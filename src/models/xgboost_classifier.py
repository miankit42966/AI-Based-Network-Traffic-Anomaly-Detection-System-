from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


@dataclass(slots=True)
class GradientModel:
    random_state: int = 42
    model: object = field(init=False)
    backend: str = field(init=False)
    num_classes_: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.backend = 'xgboost' if XGBClassifier is not None else 'sklearn_fallback'
        self.model = None

    def _build_model(self, num_classes: int) -> object:
        if XGBClassifier is not None:
            common = dict(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss' if num_classes <= 2 else 'mlogloss',
                random_state=self.random_state,
            )
            if num_classes <= 2:
                return XGBClassifier(objective='binary:logistic', **common)
            return XGBClassifier(objective='multi:softprob', num_class=num_classes, **common)
        return HistGradientBoostingClassifier(random_state=self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientModel':
        self.num_classes_ = int(len(np.unique(y)))
        self.model = self._build_model(self.num_classes_)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                return probabilities[:, 1]
            return probabilities

        raw = self.model.predict(X)
        return raw.astype(float)

    def predict_attack_score(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        if getattr(probabilities, 'ndim', 1) == 1:
            return probabilities
        return np.max(probabilities, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        if getattr(probabilities, 'ndim', 1) == 1:
            return (probabilities >= 0.5).astype(int)
        return np.argmax(probabilities, axis=1)
