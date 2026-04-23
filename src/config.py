from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    raw_data: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "raw")
    processed_data: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "processed")
    models_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "processed" / "models")


@dataclass(slots=True)
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    apply_smote: bool = True
    anomaly_percentile: int = 95
    sequence_length: int = 8
    ensemble_threshold: float = 0.75


PATHS = Paths()
TRAINING_CONFIG = TrainingConfig()
