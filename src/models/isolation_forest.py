from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass(slots=True)
class IsolationForestModel:
    contamination: float = 0.05
    random_state: int = 42
    model: IsolationForest = field(init=False)

    def __post_init__(self) -> None:
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=250,
        )

    def fit(self, X: np.ndarray) -> "IsolationForestModel":
        self.model.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        scores = -self.model.score_samples(X)
        return _normalize(scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.score_samples(X) >= 0.5).astype(int)


def _normalize(values: np.ndarray) -> np.ndarray:
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value == 0:
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)
