from __future__ import annotations

from src.data_loader import generate_synthetic_dataset
from src.models.isolation_forest import IsolationForestModel
from src.models.xgboost_classifier import GradientModel
from src.preprocessing import preprocess_dataset


def test_supervised_and_isolation_models_train_and_score() -> None:
    frame = generate_synthetic_dataset(rows=300, features=12)
    processed = preprocess_dataset(frame)

    isolation_model = IsolationForestModel().fit(processed.X_train)
    classifier_model = GradientModel().fit(processed.X_train, processed.y_train)

    iso_scores = isolation_model.score_samples(processed.X_test[:10])
    clf_scores = classifier_model.predict_proba(processed.X_test[:10])

    assert len(iso_scores) == 10
    assert len(clf_scores) == 10
    assert ((clf_scores >= 0.0) & (clf_scores <= 1.0)).all()
