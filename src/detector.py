from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.alerting import Alert
from src.feature_extractor import extract_flow_features
from src.preprocessing import build_sequences


@dataclass(slots=True)
class DetectionResult:
    score: float
    congestion_score: float
    alerts: list[dict[str, Any]]


@dataclass(slots=True)
class TrafficAnalyzer:
    isolation_model: Any
    classifier_model: Any
    lstm_model: Any | None
    preprocessor: Any
    input_columns: list[str]
    threshold: float = 0.75
    congestion_threshold: float = 80.0
    sequence_length: int = 8
    flow_buffer: deque = field(default_factory=lambda: deque(maxlen=512))
    score_buffer: deque = field(default_factory=lambda: deque(maxlen=512))

    def extract_features(self, packet: dict[str, Any]) -> np.ndarray:
        feature_vector = extract_flow_features(packet)
        raw_frame = pd.DataFrame([feature_vector.values]).reindex(columns=self.input_columns, fill_value=0.0)
        transformed = self.preprocessor.transform(raw_frame)
        if hasattr(transformed, 'toarray'):
            transformed = transformed.toarray()
        return transformed

    def detect(self, packet: dict[str, Any]) -> DetectionResult:
        transformed = self.extract_features(packet)
        isolation_score = float(self.isolation_model.score_samples(transformed)[0])
        supervised_score = float(self.classifier_model.predict_attack_score(transformed)[0])
        scores = [isolation_score, supervised_score]

        if self.lstm_model is not None and getattr(self.lstm_model, 'is_available', False):
            self.flow_buffer.append(transformed[0])
            sequences = build_sequences(np.asarray(self.flow_buffer), self.sequence_length)
            if len(sequences) > 0:
                lstm_score = float(self.lstm_model.score_samples(sequences)[-1])
                scores.append(lstm_score)
        else:
            self.flow_buffer.append(transformed[0])

        ensemble_score = float(np.mean(scores))
        self.score_buffer.append(ensemble_score)

        alerts: list[dict[str, Any]] = []
        if ensemble_score >= self.threshold:
            alerts.append(Alert.build('anomaly', ensemble_score, 'Anomalous traffic pattern detected').as_dict())

        congestion_score = self.monitor_congestion()
        if congestion_score >= 1.0:
            alerts.append(Alert.build('congestion', min(congestion_score, 1.0), 'Traffic congestion threshold exceeded').as_dict())

        return DetectionResult(
            score=ensemble_score,
            congestion_score=congestion_score,
            alerts=alerts,
        )

    def monitor_congestion(self, window: int = 60) -> float:
        if not self.flow_buffer:
            return 0.0
        packets_per_second = len(list(self.flow_buffer)[-window:]) / max(window, 1)
        return packets_per_second / self.congestion_threshold
