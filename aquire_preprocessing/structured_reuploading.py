"""Fold-local clinical concept coordinates for structured quantum re-uploading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, RobustScaler


@dataclass(frozen=True)
class ClinicalConceptGroup:
    name: str
    anchor_feature: str
    features: tuple[str, ...]


def load_structured_reuploading_config(
    path: Path, approved_features: Iterable[str]
) -> tuple[dict, tuple[ClinicalConceptGroup, ...]]:
    payload = json.loads(Path(path).read_text())
    raw_groups = payload.get("concept_groups", [])
    if len(raw_groups) != 4:
        raise ValueError("Structured re-uploading requires exactly four concept groups")
    approved = set(approved_features)
    groups = tuple(
        ClinicalConceptGroup(
            name=str(group["name"]),
            anchor_feature=str(group["anchor_feature"]),
            features=tuple(str(value) for value in group["features"]),
        )
        for group in raw_groups
    )
    names = [group.name for group in groups]
    if len(names) != len(set(names)):
        raise ValueError("Concept group names must be unique")
    used: list[str] = []
    for group in groups:
        if not group.features or len(group.features) != len(set(group.features)):
            raise ValueError(f"Invalid features in concept group {group.name}")
        if group.anchor_feature not in group.features:
            raise ValueError(f"Anchor is outside concept group {group.name}")
        missing = set(group.features) - approved
        if missing:
            raise ValueError(f"Unapproved concept features in {group.name}: {sorted(missing)}")
        used.extend(group.features)
    duplicates = sorted({name for name in used if used.count(name) > 1})
    if duplicates:
        raise ValueError(f"Concept groups must be disjoint: {duplicates}")
    return payload, groups


class FoldLocalConceptEncoder:
    """Compress four clinical families without access to disease labels.

    Every imputer, scaler, PCA and quantile transform is fitted on the outer
    training partition. PCA sign is fixed by a prespecified anchor loading,
    rather than its association with the target.
    """

    def __init__(self, groups: tuple[ClinicalConceptGroup, ...], seed: int = 42):
        if len(groups) != 4:
            raise ValueError("Exactly four concept groups are required")
        self.groups = groups
        self.seed = int(seed)
        self.transforms_: list[dict] | None = None

    def fit(self, values: np.ndarray, columns: Iterable[str]) -> "FoldLocalConceptEncoder":
        array = np.asarray(values, dtype=float)
        columns = tuple(columns)
        if array.ndim != 2 or array.shape[1] != len(columns):
            raise ValueError("Clinical matrix and columns disagree")
        if len(columns) != len(set(columns)):
            raise ValueError("Clinical columns must be unique")
        column_index = {name: index for index, name in enumerate(columns)}
        fitted = []
        for group_index, group in enumerate(self.groups):
            indices = np.asarray([column_index[name] for name in group.features], dtype=int)
            raw = array[:, indices]
            imputer = SimpleImputer(strategy="median")
            scaler = RobustScaler(quantile_range=(25.0, 75.0))
            filled = imputer.fit_transform(raw)
            scaled = scaler.fit_transform(filled)
            pca = PCA(n_components=1, svd_solver="full")
            coordinate = pca.fit_transform(scaled).reshape(-1, 1)
            anchor_index = group.features.index(group.anchor_feature)
            sign = 1.0 if float(pca.components_[0, anchor_index]) >= 0.0 else -1.0
            coordinate *= sign
            quantile = QuantileTransformer(
                n_quantiles=min(256, len(coordinate)),
                output_distribution="uniform",
                random_state=self.seed + group_index,
            )
            quantile.fit(coordinate)
            fitted.append(
                {
                    "group": group,
                    "indices": indices,
                    "imputer": imputer,
                    "scaler": scaler,
                    "pca": pca,
                    "sign": sign,
                    "quantile": quantile,
                }
            )
        self.transforms_ = fitted
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.transforms_ is None:
            raise RuntimeError("Concept encoder has not been fitted")
        array = np.asarray(values, dtype=float)
        if array.ndim != 2:
            raise ValueError("Clinical matrix must be two-dimensional")
        coordinates = []
        for fitted in self.transforms_:
            raw = array[:, fitted["indices"]]
            scaled = fitted["scaler"].transform(fitted["imputer"].transform(raw))
            coordinate = fitted["pca"].transform(scaled) * fitted["sign"]
            angle = (
                2.0 * fitted["quantile"].transform(coordinate) - 1.0
            ) * (np.pi / 2.0)
            coordinates.append(angle.reshape(-1))
        result = np.column_stack(coordinates).astype(np.float32)
        if result.shape != (len(array), 4) or not np.isfinite(result).all():
            raise RuntimeError("Invalid structured clinical coordinates")
        return result

    def fit_transform(self, values: np.ndarray, columns: Iterable[str]) -> np.ndarray:
        return self.fit(values, columns).transform(values)

    def audit(self) -> list[dict]:
        if self.transforms_ is None:
            raise RuntimeError("Concept encoder has not been fitted")
        return [
            {
                "name": fitted["group"].name,
                "anchor_feature": fitted["group"].anchor_feature,
                "features": list(fitted["group"].features),
                "feature_count": len(fitted["group"].features),
                "pca_explained_variance_ratio": float(
                    fitted["pca"].explained_variance_ratio_[0]
                ),
                "orientation_sign": float(fitted["sign"]),
            }
            for fitted in self.transforms_
        ]
