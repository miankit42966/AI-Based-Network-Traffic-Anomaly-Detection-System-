from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from tensorflow.keras.layers import Dense, Input, LSTM, RepeatVector, TimeDistributed
    from tensorflow.keras.models import Model
except Exception:
    Dense = Input = LSTM = RepeatVector = TimeDistributed = Model = None


@dataclass(slots=True)
class LSTMAutoencoderModel:
    timesteps: int
    features: int
    latent_units: int = 64
    epochs: int = 5
    batch_size: int = 64
    model: Model | None = field(init=False)
    threshold_: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.model = self._build() if Model is not None else None
        self.threshold_ = None

    def _build(self) -> Model | None:
        if Model is None:
            return None

        inputs = Input(shape=(self.timesteps, self.features))
        encoded = LSTM(self.latent_units, activation="relu", return_sequences=False)(inputs)
        repeated = RepeatVector(self.timesteps)(encoded)
        decoded = LSTM(self.latent_units, activation="relu", return_sequences=True)(repeated)
        outputs = TimeDistributed(Dense(self.features))(decoded)
        model = Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def fit(self, X_sequences: np.ndarray) -> "LSTMAutoencoderModel":
        if not self.is_available or len(X_sequences) == 0:
            return self

        self.model.fit(
            X_sequences,
            X_sequences,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            validation_split=0.1,
        )
        errors = self.reconstruction_error(X_sequences)
        self.threshold_ = float(np.percentile(errors, 95))
        return self

    def reconstruction_error(self, X_sequences: np.ndarray) -> np.ndarray:
        if not self.is_available or len(X_sequences) == 0:
            return np.zeros(len(X_sequences))

        reconstructed = self.model.predict(X_sequences, verbose=0)
        return np.mean(np.power(X_sequences - reconstructed, 2), axis=(1, 2))

    def score_samples(self, X_sequences: np.ndarray) -> np.ndarray:
        errors = self.reconstruction_error(X_sequences)
        if len(errors) == 0:
            return errors
        denominator = max(float(np.max(errors) - np.min(errors)), 1e-9)
        return (errors - np.min(errors)) / denominator
