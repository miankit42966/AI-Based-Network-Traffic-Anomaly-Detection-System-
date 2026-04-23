from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.config import PATHS, TRAINING_CONFIG
from src.data_loader import generate_synthetic_dataset, load_dataset, save_processed_dataset
from src.detector import TrafficAnalyzer
from src.feature_extractor import extract_from_dataframe
from src.models.isolation_forest import IsolationForestModel
from src.models.lstm_autoencoder import LSTMAutoencoderModel
from src.models.xgboost_classifier import GradientModel
from src.preprocessing import PreprocessedData, build_sequences, preprocess_dataset

MAX_LSTM_ELEMENTS = 50_000_000


@dataclass(slots=True)
class PipelineArtifacts:
    raw_frame: pd.DataFrame
    featured_frame: pd.DataFrame
    processed: PreprocessedData
    isolation_model: IsolationForestModel
    classifier_model: GradientModel
    lstm_model: LSTMAutoencoderModel | None
    metrics: dict[str, Any]
    alerts_preview: list[dict[str, Any]] = field(default_factory=list)


def phase_1_collect(data_path: str | Path | None = None, synthetic_rows: int = 5000, max_rows: int | None = None) -> pd.DataFrame:
    if data_path:
        bundle = load_dataset(data_path, max_rows=max_rows)
        return bundle.frame
    return generate_synthetic_dataset(rows=synthetic_rows)


def phase_2_preprocess(frame: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessedData]:
    featured = extract_from_dataframe(frame)
    processed = preprocess_dataset(featured)
    return featured, processed


def phase_3_train(processed: PreprocessedData) -> tuple[IsolationForestModel, GradientModel, LSTMAutoencoderModel | None]:
    isolation_model = IsolationForestModel().fit(processed.X_train)
    classifier_model = GradientModel().fit(processed.X_train, processed.y_train)

    train_sequences = build_sequences(processed.X_train, TRAINING_CONFIG.sequence_length)
    lstm_model: LSTMAutoencoderModel | None = None
    if train_sequences.size > 0 and train_sequences.size <= MAX_LSTM_ELEMENTS:
        candidate = LSTMAutoencoderModel(
            timesteps=TRAINING_CONFIG.sequence_length,
            features=processed.X_train.shape[1],
        )
        candidate.fit(train_sequences)
        if candidate.is_available:
            lstm_model = candidate

    return isolation_model, classifier_model, lstm_model


def phase_4_detector(
    processed: PreprocessedData,
    isolation_model: IsolationForestModel,
    classifier_model: GradientModel,
    lstm_model: LSTMAutoencoderModel | None,
) -> tuple[TrafficAnalyzer, list[dict[str, Any]]]:
    analyzer = TrafficAnalyzer(
        isolation_model=isolation_model,
        classifier_model=classifier_model,
        lstm_model=lstm_model,
        preprocessor=processed.preprocessor,
        input_columns=processed.input_columns,
        threshold=TRAINING_CONFIG.ensemble_threshold,
        sequence_length=TRAINING_CONFIG.sequence_length,
    )

    sample_alerts: list[dict[str, Any]] = []
    sample_packets = [
        {'size': 1500, 'duration': 1.2, 'ttl': 32, 'protocol': 6, 'src_port': 44321, 'dst_port': 80, 'flags': 'S', 'packets': 40, 'bytes': 120000},
        {'size': 64, 'duration': 0.2, 'ttl': 128, 'protocol': 17, 'src_port': 53000, 'dst_port': 53, 'flags': 'A', 'packets': 4, 'bytes': 256},
    ]
    for packet in sample_packets:
        result = analyzer.detect(packet)
        sample_alerts.extend(result.alerts)

    return analyzer, sample_alerts


def _compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if getattr(y_prob, 'ndim', 1) == 1:
        return float(roc_auc_score(y_true, y_prob))
    return float(roc_auc_score(y_true, y_prob, multi_class='ovr'))


def phase_6_evaluate(
    processed: PreprocessedData,
    isolation_model: IsolationForestModel,
    classifier_model: GradientModel,
) -> dict[str, Any]:
    y_prob = classifier_model.predict_proba(processed.X_test)
    y_pred = classifier_model.predict(processed.X_test)

    latencies = []
    for row in processed.X_test[: min(256, len(processed.X_test))]:
        start = time.perf_counter()
        classifier_model.predict(row.reshape(1, -1))
        latencies.append(time.perf_counter() - start)

    metrics = {
        'f1': float(f1_score(processed.y_test, y_pred, average='weighted')),
        'precision': float(precision_score(processed.y_test, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(processed.y_test, y_pred, average='weighted', zero_division=0)),
        'roc_auc': _compute_roc_auc(processed.y_test, y_prob),
        'confusion_matrix': confusion_matrix(processed.y_test, y_pred).tolist(),
        'classification_report': classification_report(processed.y_test, y_pred, zero_division=0),
        'latency_ms_avg': float(np.mean(latencies) * 1000) if latencies else 0.0,
        'latency_ms_p99': float(np.percentile(latencies, 99) * 1000) if latencies else 0.0,
        'isolation_score_mean': float(np.mean(isolation_model.score_samples(processed.X_test))),
    }
    return metrics


def save_artifacts(
    processed: PreprocessedData,
    isolation_model: IsolationForestModel,
    classifier_model: GradientModel,
    metrics: dict[str, Any],
) -> None:
    PATHS.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(processed.preprocessor, PATHS.models_dir / 'preprocessor.joblib')
    joblib.dump(processed.label_encoder, PATHS.models_dir / 'label_encoder.joblib')
    joblib.dump(isolation_model, PATHS.models_dir / 'isolation_forest.joblib')
    joblib.dump(classifier_model, PATHS.models_dir / 'classifier.joblib')
    (PATHS.models_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')


def run_all(data_path: str | Path | None = None, synthetic_rows: int = 5000, max_rows: int | None = None) -> PipelineArtifacts:
    raw_frame = phase_1_collect(data_path=data_path, synthetic_rows=synthetic_rows, max_rows=max_rows)
    save_processed_dataset(raw_frame, PATHS.processed_data / 'phase_1_raw_snapshot.csv')

    featured_frame, processed = phase_2_preprocess(raw_frame)
    save_processed_dataset(featured_frame, PATHS.processed_data / 'phase_2_featured_snapshot.csv')

    isolation_model, classifier_model, lstm_model = phase_3_train(processed)
    _, alerts_preview = phase_4_detector(processed, isolation_model, classifier_model, lstm_model)
    metrics = phase_6_evaluate(processed, isolation_model, classifier_model)
    save_artifacts(processed, isolation_model, classifier_model, metrics)

    return PipelineArtifacts(
        raw_frame=raw_frame,
        featured_frame=featured_frame,
        processed=processed,
        isolation_model=isolation_model,
        classifier_model=classifier_model,
        lstm_model=lstm_model,
        metrics=metrics,
        alerts_preview=alerts_preview,
    )
