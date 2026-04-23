from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.config import TRAINING_CONFIG

HIGH_CARDINALITY_COLUMNS = {
    'timestamp',
    'flow id',
    'src ip',
    'dst ip',
    'source ip',
    'destination ip',
}


@dataclass(slots=True)
class PreprocessedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    input_columns: list[str]
    label_encoder: LabelEncoder
    preprocessor: ColumnTransformer


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = cleaned.columns.str.strip()
    cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    cleaned.dropna(axis=1, how='all', inplace=True)
    cleaned.drop_duplicates(inplace=True)
    return cleaned


def split_features_and_labels(frame: pd.DataFrame, label_column: str = 'Label') -> tuple[pd.DataFrame, pd.Series]:
    cleaned = clean_frame(frame)
    if label_column not in cleaned.columns:
        raise KeyError(f"Label column '{label_column}' is missing")

    X = cleaned.drop(columns=[label_column])
    y = cleaned[label_column].astype(str)

    removable = [column for column in X.columns if column.strip().lower() in HIGH_CARDINALITY_COLUMNS]
    if removable:
        X = X.drop(columns=removable)

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=['number', 'bool']).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_columns),
            ('cat', categorical_pipeline, categorical_columns),
        ]
    )


def preprocess_dataset(frame: pd.DataFrame, label_column: str = 'Label') -> PreprocessedData:
    X, y = split_features_and_labels(frame, label_column=label_column)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TRAINING_CONFIG.test_size,
        random_state=TRAINING_CONFIG.random_state,
        stratify=y_encoded,
    )

    preprocessor = build_preprocessor(X_train_df)
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    if hasattr(X_train, 'toarray'):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    if TRAINING_CONFIG.apply_smote and len(np.unique(y_train)) > 1:
        smote = SMOTE(random_state=TRAINING_CONFIG.random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    feature_names = preprocessor.get_feature_names_out().tolist()
    return PreprocessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        input_columns=X.columns.tolist(),
        label_encoder=label_encoder,
        preprocessor=preprocessor,
    )


def build_sequences(X: np.ndarray, sequence_length: int) -> np.ndarray:
    if len(X) < sequence_length:
        return np.empty((0, sequence_length, X.shape[1]))

    return np.stack([X[idx : idx + sequence_length] for idx in range(len(X) - sequence_length + 1)])
