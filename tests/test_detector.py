from __future__ import annotations

from src.data_loader import generate_synthetic_dataset
from src.detector import TrafficAnalyzer
from src.models.isolation_forest import IsolationForestModel
from src.models.xgboost_classifier import GradientModel
from src.preprocessing import preprocess_dataset


def test_detector_returns_score_and_congestion() -> None:
    frame = generate_synthetic_dataset(rows=300, features=10)
    processed = preprocess_dataset(frame)
    analyzer = TrafficAnalyzer(
        isolation_model=IsolationForestModel().fit(processed.X_train),
        classifier_model=GradientModel().fit(processed.X_train, processed.y_train),
        lstm_model=None,
        preprocessor=processed.preprocessor,
        input_columns=processed.input_columns,
    )

    packet = {"size": 1200, "duration": 1.5, "ttl": 56, "protocol": 6, "src_port": 51515, "dst_port": 443, "flags": "SA", "packets": 24, "bytes": 42000}
    result = analyzer.detect(packet)

    assert 0.0 <= result.score <= 1.0
    assert result.congestion_score >= 0.0
