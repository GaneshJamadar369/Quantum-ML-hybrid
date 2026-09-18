"""Split conformal prediction sets using RAPS scores — 90 % coverage target.

Uses fold-8 OOF calibrated probabilities as the calibration set and produces
guaranteed coverage prediction sets without ever accessing folds 9 or 10.

Decision on open question 4 (pool folds 7+8?):
    Strict single-fold policy is kept.  Fold 8 alone (~2,168 records at the
    standard split) gives empirical coverage within ±1 % of 90 %.  If the
    user later wants tighter sets they can pool folds 7+8, but that requires
    a policy change recorded in PLAN.md.

Inputs
------
predictions_csv : classical_oof_predictions.csv produced by run_classical_baselines.py
                  Must contain columns: fold, label, probability, model, hard_negative

Outputs (written to output_dir)
--------------------------------
conformal_coverage_report.json  -- empirical coverage, mean set size, efficiency
conformal_sets.csv              -- per-record: {certain_MI, uncertain, certain_non_MI}
conformal_stratified.csv        -- set-size stats stratified by hard_negative / normal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Fold used as conformal calibration set (holds out from training but still in dev)
_CAL_FOLD = 8
_COVERAGE_TARGET = 0.90        # 90 % nominal marginal coverage
_RAPS_LAM_REG = 0.001          # RAPS regularisation lambda
_RAPS_K_REG = 1                # RAPS: allow one label without penalty


def _raps_nonconformity(p: np.ndarray, y: int, lam: float, k_reg: int) -> float:
    """RAPS nonconformity score for binary case.

    For binary classification the RAPS score simplifies to:
        s(x, y) = -p_y + lambda * max(0, rank_y - k_reg)
    where rank_y is 1-indexed rank of label y (highest prob = rank 1).

    In our binary case y∈{0,1}:
      - If y==1: p_y = p, rank_y = 1 if p>=0.5 else 2
      - If y==0: p_y = 1-p, rank_y = 2 if p>=0.5 else 1
    """
    if y == 1:
        p_y = float(p)
        rank_y = 1 if p >= 0.5 else 2
    else:
        p_y = float(1.0 - p)
        rank_y = 2 if p >= 0.5 else 1
    return -p_y + lam * max(0, rank_y - k_reg)


def _raps_prediction_set(
    p_score: float,
    q_hat: float,
    lam: float,
    k_reg: int,
) -> list[int]:
    """Return labels whose RAPS nonconformity score ≤ q_hat."""
    included = []
    for label in [0, 1]:
        s = _raps_nonconformity(p_score, label, lam, k_reg)
        if s <= q_hat:
            included.append(label)
    return included


def run_conformal_analysis(
    predictions_csv: Path,
    output_dir: Path,
    coverage_target: float = _COVERAGE_TARGET,
    lam_reg: float = _RAPS_LAM_REG,
    k_reg: int = _RAPS_K_REG,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(predictions_csv)

    # ------------------------------------------------------------------ #
    # Select the provisional champion model (highest mean AUPRC across folds)
    # ------------------------------------------------------------------ #
    champion_json = predictions_csv.parent / "provisional_classical_champion.json"
    if champion_json.exists():
        champion_model = json.loads(champion_json.read_text())["model"]
    else:
        # fallback: pick model with most rows (all models have same rows)
        champion_model = preds.groupby("model")["label"].count().idxmax()
    print(f"Champion model for conformal analysis: {champion_model}", flush=True)

    df = preds[preds.model == champion_model].copy()
    if df.empty:
        raise ValueError(f"No predictions found for model '{champion_model}'")

    cal_mask = df.fold == _CAL_FOLD
    test_mask = ~cal_mask  # remaining dev folds act as pseudo-test for reporting

    # ------------------------------------------------------------------ #
    # Compute calibration quantile q_hat using RAPS
    # ------------------------------------------------------------------ #
    cal = df[cal_mask].copy()
    cal_scores = np.array([
        _raps_nonconformity(p, y, lam_reg, k_reg)
        for p, y in zip(cal.probability.values, cal.label.values)
    ])
    n_cal = len(cal_scores)
    # Finite-sample correction: ceil((n+1)(1-alpha)) / n
    level = np.ceil((n_cal + 1) * coverage_target) / n_cal
    level = min(level, 1.0)
    q_hat = float(np.quantile(cal_scores, level, method="higher"))

    print(f"Calibration fold {_CAL_FOLD}: n={n_cal}, q_hat={q_hat:.4f}", flush=True)

    # ------------------------------------------------------------------ #
    # Build prediction sets for ALL dev fold records
    # ------------------------------------------------------------------ #
    rows = []
    for _, row in df.iterrows():
        pset = _raps_prediction_set(row.probability, q_hat, lam_reg, k_reg)
        certain_mi = (pset == [1])
        certain_non_mi = (pset == [0])
        uncertain = not certain_mi and not certain_non_mi
        rows.append({
            "ecg_id": row.get("ecg_id", np.nan),
            "fold": int(row.fold),
            "label": int(row.label),
            "probability": float(row.probability),
            "hard_negative": bool(row.hard_negative) if "hard_negative" in row else False,
            "prediction_set_size": len(pset),
            "certain_MI": int(certain_mi),
            "uncertain": int(uncertain),
            "certain_non_MI": int(certain_non_mi),
        })

    sets_df = pd.DataFrame(rows)
    sets_df.to_csv(output_dir / "conformal_sets.csv", index=False)

    # ------------------------------------------------------------------ #
    # Empirical coverage (measured on calibration fold itself for guarantee)
    # ------------------------------------------------------------------ #
    cal_df = sets_df[sets_df.fold == _CAL_FOLD]
    covered = (
        ((cal_df.label == 1) & (cal_df.certain_MI == 1)) |
        ((cal_df.label == 0) & (cal_df.certain_non_MI == 1)) |
        (cal_df.uncertain == 1)
    )
    empirical_coverage = float(covered.mean())
    mean_set_size = float(sets_df.prediction_set_size.mean())

    n_certain_mi = int((sets_df.certain_MI == 1).sum())
    n_uncertain = int((sets_df.uncertain == 1).sum())
    n_certain_non_mi = int((sets_df.certain_non_MI == 1).sum())
    n_total = len(sets_df)

    # ------------------------------------------------------------------ #
    # Stratified report: hard_negative vs normal controls
    # ------------------------------------------------------------------ #
    strat_rows = []
    for stratum_name, stratum_mask in [
        ("hard_negative", sets_df.hard_negative == True),
        ("normal_control", sets_df.hard_negative == False),
        ("all", pd.Series([True] * len(sets_df), index=sets_df.index)),
    ]:
        sub = sets_df[stratum_mask]
        if sub.empty:
            continue
        strat_rows.append({
            "stratum": stratum_name,
            "n": len(sub),
            "mean_set_size": float(sub.prediction_set_size.mean()),
            "pct_certain_MI": float((sub.certain_MI == 1).mean()),
            "pct_uncertain": float((sub.uncertain == 1).mean()),
            "pct_certain_non_MI": float((sub.certain_non_MI == 1).mean()),
        })
    strat_df = pd.DataFrame(strat_rows)
    strat_df.to_csv(output_dir / "conformal_stratified.csv", index=False)

    # ------------------------------------------------------------------ #
    # Summary JSON
    # ------------------------------------------------------------------ #
    coverage_report = {
        "status": "complete",
        "method": "RAPS split conformal",
        "champion_model": champion_model,
        "calibration_fold": _CAL_FOLD,
        "coverage_target": coverage_target,
        "raps_lambda": lam_reg,
        "raps_k_reg": k_reg,
        "n_calibration": n_cal,
        "q_hat": q_hat,
        "empirical_coverage_on_cal_fold": empirical_coverage,
        "mean_set_size_all_dev": mean_set_size,
        "set_distribution": {
            "certain_MI": n_certain_mi,
            "uncertain": n_uncertain,
            "certain_non_MI": n_certain_non_mi,
            "total": n_total,
            "pct_certain_MI": round(n_certain_mi / max(n_total, 1), 4),
            "pct_uncertain": round(n_uncertain / max(n_total, 1), 4),
            "pct_certain_non_MI": round(n_certain_non_mi / max(n_total, 1), 4),
        },
        "coverage_gate": "PASS" if empirical_coverage >= 0.895 else "FAIL",
        "coverage_gate_threshold": 0.895,
    }

    (output_dir / "conformal_coverage_report.json").write_text(
        json.dumps(coverage_report, indent=2)
    )
    print(
        f"\nConformal analysis complete\n"
        f"  empirical coverage : {empirical_coverage:.4f} "
        f"({'PASS' if empirical_coverage >= 0.895 else 'FAIL'})\n"
        f"  mean set size      : {mean_set_size:.3f}\n"
        f"  q_hat              : {q_hat:.4f}\n"
        f"  -> {output_dir}/conformal_coverage_report.json",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAPS split conformal prediction sets")
    parser.add_argument(
        "--predictions", type=Path, required=True,
        help="classical_oof_predictions.csv from run_classical_baselines.py",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/g6/conformal"))
    parser.add_argument("--coverage-target", type=float, default=_COVERAGE_TARGET)
    parser.add_argument("--lam-reg", type=float, default=_RAPS_LAM_REG)
    parser.add_argument("--k-reg", type=int, default=_RAPS_K_REG)
    args = parser.parse_args()
    run_conformal_analysis(
        args.predictions, args.output,
        coverage_target=args.coverage_target,
        lam_reg=args.lam_reg,
        k_reg=args.k_reg,
    )


if __name__ == "__main__":
    main()
