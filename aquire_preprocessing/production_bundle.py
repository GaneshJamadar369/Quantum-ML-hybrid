"""Runtime loader for the frozen, always-active hybrid prototype bundle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from .feature_evidence import derive_clinical_composites
from .features import extract_deployable_features
from .models_quantum import TorchStatevectorQuantumClassifier
from .models_transformer import ECGPatchTransformer
from .prototype_bundle import VerifiedBundle


def _single(bundle: VerifiedBundle, role: str) -> Path:
    paths = bundle.paths_for(role)
    if len(paths) != 1:
        raise ValueError(f"Expected one artifact for role {role!r}, found {len(paths)}")
    return paths[0]


def _torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location="cpu")


class FrozenHybridBundle:
    """Execute Transformer/PLS/VQC and morphology/HGB before fixed fusion.

    The class deliberately has no branch-selection or fallback method. A
    failure in either branch aborts the prediction.
    """

    def __init__(self, bundle: VerifiedBundle, *, device: str = "cpu") -> None:
        import joblib
        import sklearn
        import torch

        self.bundle = bundle
        expected_sklearn = bundle.manifest.get("artifact_runtime", {}).get("scikit_learn")
        if expected_sklearn and sklearn.__version__ != expected_sklearn:
            raise RuntimeError(
                f"Bundle requires scikit-learn {expected_sklearn}; runtime has {sklearn.__version__}"
            )
        self.device = torch.device(device)
        self.transformer = ECGPatchTransformer().to(self.device)
        self.transformer.load_state_dict(_torch_load(_single(bundle, "transformer"))["state_dict"])
        self.transformer.eval()

        self.normalizer = json.loads(_single(bundle, "waveform_normalizer").read_text())
        self.h128_imputer = joblib.load(_single(bundle, "h128_imputer"))
        self.h128_scaler = joblib.load(_single(bundle, "h128_scaler"))
        self.pls_payload = joblib.load(_single(bundle, "pls_q4"))
        self.angle_quantiles = joblib.load(_single(bundle, "angle_quantiles"))
        self.quantum_alignment = joblib.load(_single(bundle, "vqc_score_alignment"))

        self.vqcs = []
        for path in bundle.paths_for("vqc_model"):
            payload = _torch_load(path)
            model = TorchStatevectorQuantumClassifier(4, n_layers=2, topology="ring").to(self.device)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            self.vqcs.append(model)

        self.feature_manifest = json.loads(_single(bundle, "morphology_feature_manifest").read_text())
        self.morphology_conditioner = joblib.load(_single(bundle, "morphology_conditioner"))
        self.morphology_hgb = joblib.load(_single(bundle, "morphology_hgb"))
        self.fusion = json.loads(_single(bundle, "fusion").read_text())
        self.calibrator = joblib.load(_single(bundle, "platt_calibrator"))
        self.threshold = float(json.loads(_single(bundle, "decision_threshold").read_text())["threshold"])

    def _normalise(self, signal: np.ndarray) -> np.ndarray:
        medians = np.asarray([self.normalizer["lead_medians"][lead] for lead in self.normalizer["lead_order"]])
        iqrs = np.asarray([self.normalizer["lead_iqrs"][lead] for lead in self.normalizer["lead_order"]])
        return ((signal - medians[:, None]) / iqrs[:, None]).astype(np.float32)

    def quantum_score(self, signal: np.ndarray) -> float:
        import torch

        normalized = self._normalise(signal)[None]
        with torch.inference_mode():
            _, representation = self.transformer(torch.from_numpy(normalized).to(self.device))
        h128 = representation.cpu().numpy()
        x = self.h128_scaler.transform(self.h128_imputer.transform(h128))
        q4 = self.pls_payload["model"].transform(x) * np.asarray(self.pls_payload["signs"])
        q4 = (2.0 * self.angle_quantiles.transform(q4) - 1.0) * (np.pi / 2.0)
        tensor = torch.from_numpy(q4.astype(np.float32)).to(self.device)
        cdf_scores = []
        with torch.inference_mode():
            for model, reference in zip(self.vqcs, self.quantum_alignment["train_reference_logits"]):
                raw = float(model(tensor).cpu().numpy()[0])
                ordered = np.asarray(reference, dtype=float)
                cdf_scores.append((np.searchsorted(ordered, raw, side="right") + 0.5) / (len(ordered) + 1.0))
        mean_cdf = float(np.mean(cdf_scores))
        return float(self.quantum_alignment["calibrator"].predict_proba([[mean_cdf]])[0, 1])

    def clinical_feature_values(self, signal: np.ndarray) -> tuple[dict[str, float], list[str]]:
        """Return the deployable morphology values used by the clinical route.

        The public prototype uses this same extraction path to present a small,
        clinician-readable signal summary beside the screening result.  No
        label-derived or PTB-XL+ commercial measurement is introduced here.
        """
        bundle = extract_deployable_features(signal, 100, ecg_id=-1)
        base = pd.DataFrame([bundle.values])
        derived, _ = derive_clinical_composites(base)
        complete = pd.concat([base, derived], axis=1)
        values = {
            str(name): float(value) if pd.notna(value) else float("nan")
            for name, value in complete.iloc[0].items()
        }
        return values, list(bundle.failures)

    def classical_score(self, signal: np.ndarray) -> float:
        values, _ = self.clinical_feature_values(signal)
        complete = pd.DataFrame([values])
        approved = self.feature_manifest["approved_features"]
        missing = sorted(set(approved) - set(complete.columns))
        if missing:
            raise RuntimeError(f"Deployable extractor omitted approved features: {missing}")
        # The frozen imputer was fitted on a NumPy matrix in the exporter. Keep
        # the identical column order while avoiding a feature-name contract it
        # never learned.
        feature_matrix = complete.loc[:, approved].to_numpy(dtype=float)
        conditioned = self.morphology_conditioner["imputer"].transform(feature_matrix)
        raw_probability = self.morphology_hgb.predict_proba(conditioned)[:, 1]
        raw_logit = logit(np.clip(raw_probability, 1e-6, 1.0 - 1e-6)).reshape(-1, 1)
        return float(self.morphology_conditioner["calibrator"].predict_proba(raw_logit)[0, 1])

    def fusion_score(self, s_q: float, s_c: float) -> tuple[float, float]:
        x = np.asarray([logit(np.clip(s_c, 1e-6, 1 - 1e-6)), logit(np.clip(s_q, 1e-6, 1 - 1e-6))])
        standardized = (x - np.asarray(self.fusion["mean"])) / np.asarray(self.fusion["scale"])
        raw = float(self.fusion["intercept"] + standardized @ np.asarray(self.fusion["weights"]))
        return raw, float(expit(raw))

    def calibrate(self, fused_probability: float) -> float:
        value = logit(np.clip(fused_probability, 1e-6, 1.0 - 1e-6))
        return float(self.calibrator.predict_proba([[value]])[0, 1])

    def golden_self_test(self, *, tolerance: float = 2e-4) -> dict[str, float | int]:
        """Run the signed Fold-9 fixtures through the complete two-route path."""

        fixture = np.load(_single(self.bundle, "golden_fixture"), allow_pickle=False)
        expected = np.asarray(fixture["probability"], dtype=float)
        actual = []
        for signal in fixture["signal_mv"]:
            s_q = self.quantum_score(signal)
            s_c = self.classical_score(signal)
            _, fused = self.fusion_score(s_q, s_c)
            actual.append(self.calibrate(fused))
        error = np.abs(np.asarray(actual) - expected)
        maximum = float(error.max(initial=0.0))
        if maximum > tolerance:
            raise RuntimeError(
                f"Golden hybrid parity failed: max_abs_error={maximum:.8g} > {tolerance:.8g}"
            )
        return {"cases": int(len(expected)), "max_abs_error": maximum, "tolerance": tolerance}
