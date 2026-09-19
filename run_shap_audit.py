"""Post-OOF SHAP feature attribution audit for the provisional classical champion.

Runs on fold-8 held-out records only (never folds 9/10) using the model
retrained on folds 1-7. Restricted to tree-based models (RF, XGBoost,
HistGB) using TreeExplainer for exact, fast Shapley values, and
LinearExplainer for logistic regression. MLP/SVC are skipped.

Outputs (written to output_dir):
  shap_mean_abs_values.csv       -- global |SHAP| ranking of all features
  shap_beeswarm.png              -- beeswarm of top-20 features
  shap_dependence_top5/          -- SHAP dependence plots for top-5 features
  shap_interaction_matrix.csv    -- SHAP pairwise interaction strengths
                                    (tree models only, fold-8 model)
  shap_clinical_audit.json       -- clinical plausibility check:
                                    flags if any non-ECG feature ranks top-10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aquire_preprocessing.baselines import _models, FoldLocalPlattCalibrator
from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.feature_conditioning import FoldLocalOutlierClipper
from aquire_preprocessing.feature_interactions import ClinicalInteractionEngineer
from aquire_preprocessing.feature_manifest import load_feature_manifest
from aquire_preprocessing.features import FoldLocalTabularTransformer

# Fold used as held-out for the SHAP evaluation run
_SHAP_EVAL_FOLD = 8
# Folds used to train the model whose weights are audited
_SHAP_TRAIN_FOLDS = [f for f in DEV_FOLDS if f != _SHAP_EVAL_FOLD]

# Clinical keywords expected in the top-10 SHAP features
_CLINICAL_KEYWORDS = [
    "st", "r_amp", "rs_ratio", "rr", "clinical", "interaction",
    "t_polarity", "heart_rate", "r_mean", "s_amp",
]

# Tree-based model names that support TreeExplainer
_TREE_MODELS = {"random_forest", "hist_gradient_boosting", "xgboost"}
_LINEAR_MODELS = {"logistic"}


def _is_clinical(feature_name: str) -> bool:
    name = feature_name.lower()
    return any(kw in name for kw in _CLINICAL_KEYWORDS)


def _build_fold_matrices(
    x: pd.DataFrame,
    labels: np.ndarray,
    folds: np.ndarray,
    hard_negative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Replicate the pre-modelling pipeline for the SHAP audit fold."""
    train_mask = np.isin(folds, _SHAP_TRAIN_FOLDS)

    clipper = FoldLocalOutlierClipper()
    clipper.fit(x.loc[train_mask])
    x_clipped = clipper.transform(x)

    interactor = ClinicalInteractionEngineer()
    interactor.fit(x_clipped.loc[train_mask], labels[train_mask])
    x_inter = interactor.transform(x_clipped)

    transformer = FoldLocalTabularTransformer().fit(
        x_inter, folds, _SHAP_TRAIN_FOLDS, labels=labels, max_features=256
    )
    X_all = transformer.transform(x_inter)
    feature_names = transformer.selected_columns_

    return X_all, folds, feature_names


def run_shap_audit(
    features_csv: Path,
    metadata_csv: Path,
    manifest_path: Path,
    output_dir: Path,
    seed: int = 42,
) -> None:
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("shap and matplotlib are required: pip install shap matplotlib") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dep_dir = output_dir / "shap_dependence_top5"
    dep_dir.mkdir(exist_ok=True)

    features_raw = pd.read_csv(features_csv).set_index("ecg_id")
    approved = load_feature_manifest(manifest_path, features_raw.columns)["approved_features"]
    features_raw = features_raw[approved]

    metadata = pd.read_csv(metadata_csv).set_index("ecg_id")
    joined = metadata.join(features_raw, how="inner", validate="one_to_one")
    joined = joined[joined.strat_fold.isin(DEV_FOLDS) & joined.eligibility.eq("PRIMARY")]

    labels = joined.mi_label.to_numpy(dtype=int)
    folds = joined.strat_fold.to_numpy(dtype=int)
    hard_neg = (
        joined.hard_negative.to_numpy(dtype=bool)
        if "hard_negative" in joined.columns
        else np.zeros(len(joined), dtype=bool)
    )

    x = joined[approved].select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    X_all, folds_arr, feature_names = _build_fold_matrices(x, labels, folds, hard_neg)

    train_mask = np.isin(folds_arr, _SHAP_TRAIN_FOLDS)
    eval_mask = folds_arr == _SHAP_EVAL_FOLD
    X_train = X_all[train_mask]
    y_train = labels[train_mask]
    X_eval = X_all[eval_mask]

    audit_results = {}
    all_shap_mean_abs = {}

    for model_name, template in _models(seed).items():
        if model_name not in _TREE_MODELS and model_name not in _LINEAR_MODELS:
            print(f"  Skipping {model_name} (KernelExplainer too slow for Phase 6)")
            continue

        print(f"  Computing SHAP for {model_name} ...", flush=True)
        # Fit on train folds only
        template.fit(X_train, y_train)

        # Get the final estimator from the sklearn Pipeline
        final_estimator = template.steps[-1][1] if hasattr(template, "steps") else template

        try:
            eval_sample = X_eval[:500]
            if model_name in _TREE_MODELS:
                explainer = shap.TreeExplainer(final_estimator, feature_names=feature_names)
                shap_values = explainer.shap_values(eval_sample)
                # For binary classifiers some versions return list [neg, pos]
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:  # linear
                # Use training data summary as background
                background = shap.sample(X_train, min(200, len(X_train)), random_state=seed)
                explainer = shap.LinearExplainer(
                    final_estimator, background, feature_names=feature_names
                )
                shap_values = explainer.shap_values(eval_sample)
        except Exception as exc:
            print(f"    SHAP failed for {model_name}: {exc}")
            continue

        mean_abs = np.abs(shap_values).mean(axis=0)
        all_shap_mean_abs[model_name] = dict(zip(feature_names, mean_abs.tolist()))

        # ------------------------------------------------------------------ #
        # Beeswarm — top-20 features
        # ------------------------------------------------------------------ #
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, eval_sample,
            feature_names=feature_names,
            max_display=20, show=False, plot_size=None,
        )
        plt.tight_layout()
        fig.savefig(output_dir / f"shap_beeswarm_{model_name}.png", dpi=150)
        plt.close(fig)

        # ------------------------------------------------------------------ #
        # Dependence plots — top-5 features
        # ------------------------------------------------------------------ #
        top5_idx = np.argsort(mean_abs)[-5:][::-1]
        for i, idx in enumerate(top5_idx):
            idx_int = int(idx)
            feat = feature_names[idx_int]
            fig, ax = plt.subplots(figsize=(6, 4))
            shap.dependence_plot(
                idx_int, shap_values, eval_sample,
                feature_names=feature_names, ax=ax, show=False,
            )
            ax.set_title(f"{model_name} | {feat}")
            plt.tight_layout()
            fig.savefig(dep_dir / f"{model_name}_dep_{i + 1}_{feat[:40]}.png", dpi=120)
            plt.close(fig)

        # ------------------------------------------------------------------ #
        # SHAP interaction matrix (XGBoost champion only for speed)
        # ------------------------------------------------------------------ #
        if model_name == "xgboost":
            try:
                inter_values = explainer.shap_interaction_values(eval_sample[:100])
                if isinstance(inter_values, list):
                    inter_values = inter_values[1]
                inter_mean = np.abs(inter_values).mean(axis=0)
                inter_df = pd.DataFrame(inter_mean, index=feature_names, columns=feature_names)
                inter_df.to_csv(output_dir / "shap_interaction_matrix.csv")
            except Exception as exc:
                print(f"    Interaction values skipped: {exc}")

        # ------------------------------------------------------------------ #
        # Clinical plausibility check
        # ------------------------------------------------------------------ #
        ranked = sorted(zip(feature_names, mean_abs.tolist()), key=lambda t: t[1], reverse=True)
        top10 = [name for name, _ in ranked[:10]]
        non_clinical_top10 = [n for n in top10 if not _is_clinical(n)]
        audit_results[model_name] = {
            "top10_features": top10,
            "non_clinical_in_top10": non_clinical_top10,
            "clinical_plausibility": "PASS" if not non_clinical_top10 else "REVIEW",
            "note": (
                "All top-10 features are clinically motivated ECG measurements."
                if not non_clinical_top10
                else f"Non-clinical features in top-10: {non_clinical_top10}. Investigate for leakage."
            ),
        }

    # ------------------------------------------------------------------ #
    # Global |SHAP| CSV  (average of per-model mean-abs across all models)
    # ------------------------------------------------------------------ #
    all_features_union = sorted({f for vals in all_shap_mean_abs.values() for f in vals})
    rows = []
    for feat in all_features_union:
        scores = [v.get(feat, 0.0) for v in all_shap_mean_abs.values()]
        rows.append({"feature": feat, "mean_abs_shap": float(np.mean(scores))})
    shap_df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(output_dir / "shap_mean_abs_values.csv", index=False)

    with open(output_dir / "shap_clinical_audit.json", "w") as fh:
        json.dump(
            {
                "status": "complete",
                "eval_fold": _SHAP_EVAL_FOLD,
                "train_folds": _SHAP_TRAIN_FOLDS,
                "models_audited": list(audit_results.keys()),
                "per_model": audit_results,
            },
            fh,
            indent=2,
        )
    print(f"\nSHAP audit complete -> {output_dir}/shap_clinical_audit.json", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-OOF SHAP attribution audit")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--approved-feature-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/g6/shap"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_shap_audit(args.features, args.metadata, args.approved_feature_manifest, args.output, args.seed)


if __name__ == "__main__":
    main()
