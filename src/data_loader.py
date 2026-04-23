from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


SUPPORTED_SUFFIXES = {".csv", ".parquet"}


@dataclass(slots=True)
class DatasetBundle:
    frame: pd.DataFrame
    source: str
    label_column: str = "Label"


def load_dataset(path: str | Path, label_column: str = "Label", max_rows: int | None = None, random_state: int = 42) -> DatasetBundle:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported dataset format: {suffix}")

    if suffix == ".csv":
        frame = pd.read_csv(dataset_path)
    else:
        frame = pd.read_parquet(dataset_path)

    frame.columns = frame.columns.str.strip()
    if label_column not in frame.columns:
        raise KeyError(f"Label column '{label_column}' not found in {dataset_path}")

    if max_rows is not None and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=random_state).reset_index(drop=True)

    return DatasetBundle(frame=frame, source=str(dataset_path), label_column=label_column)


def load_multiple(paths: Iterable[str | Path], label_column: str = "Label") -> pd.DataFrame:
    frames = [load_dataset(path, label_column=label_column).frame for path in paths]
    return pd.concat(frames, ignore_index=True)


def generate_synthetic_dataset(
    rows: int = 5000,
    features: int = 24,
    attack_ratio: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    weights = [1 - attack_ratio, attack_ratio]
    X, y = make_classification(
        n_samples=rows,
        n_features=features,
        n_informative=max(4, features // 3),
        n_redundant=max(2, features // 6),
        n_classes=2,
        weights=weights,
        random_state=random_state,
    )

    columns = [f"flow_feature_{idx:02d}" for idx in range(features)]
    frame = pd.DataFrame(X, columns=columns)
    frame["Flow Duration"] = np.abs(frame["flow_feature_00"] * 120).round(2)
    frame["Flow Bytes/s"] = np.abs(frame["flow_feature_01"] * 1000).round(2)
    frame["Flow Packets/s"] = np.abs(frame["flow_feature_02"] * 100).round(2)
    frame["Destination Port"] = np.random.default_rng(random_state).integers(1, 65535, size=rows)
    frame["Protocol"] = np.random.default_rng(random_state + 1).choice([6, 17, 1], size=rows, p=[0.65, 0.3, 0.05])
    frame["Packet Length Mean"] = np.abs(frame["flow_feature_03"] * 512).round(2)
    frame["Label"] = np.where(y == 0, "BENIGN", "ATTACK")
    return frame


def save_processed_dataset(frame: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path
