# Network Anomaly Detection

An end-to-end network anomaly detection project with:

- Dataset ingestion for CICIDS-style CSV files
- Preprocessing, scaling, and imbalance handling
- Multi-model training with `IsolationForest`, LSTM autoencoder, and gradient-boosted classifier
- Real-time scoring and congestion monitoring
- Streamlit dashboard with phase-wise execution view
- Basic test coverage for the core pipeline

## Project Structure

```text
network_anomaly_detection/
+-- data/
¦   +-- raw/
¦   +-- processed/
+-- notebooks/
+-- src/
¦   +-- models/
¦   +-- alerting.py
¦   +-- cli.py
¦   +-- config.py
¦   +-- data_loader.py
¦   +-- detector.py
¦   +-- feature_extractor.py
¦   +-- pipeline.py
¦   +-- preprocessing.py
+-- dashboard/
¦   +-- app.py
+-- tests/
+-- requirements.txt
+-- README.md
```

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the end-to-end phase pipeline with a raw CSV:

```bash
python -m src.cli run-all --data-path data/raw/CICIDS2017_sample.csv
```

4. Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## Pipeline Phases

### Phase 1: Data Collection

- Load a CICIDS/UNSW/KDD-style CSV using `src.data_loader`
- Optional synthetic dataset generation for local testing
- Optional live packet capture hooks for `pyshark` and `scapy`

### Phase 2: Preprocessing & Feature Engineering

- Column cleanup
- Missing value handling
- Label encoding
- Standard scaling
- Optional SMOTE on the training split

### Phase 3: AI Model Training

- `IsolationForest` for unsupervised anomaly scoring
- LSTM autoencoder for reconstruction-error based anomaly scoring
- XGBoost classifier with a safe sklearn fallback if XGBoost is unavailable
- Weighted ensemble fusion

### Phase 4: Real-Time Detection

- Sliding window score aggregation
- Congestion estimation using packets-per-second
- Alert generation with severity bands

### Phase 5: Monitoring Dashboard

- Live metrics
- Flow trend chart
- Anomaly scatter overlay
- Alert table
- Phase flow diagram matching the requested execution layout

### Phase 6: Testing & Evaluation

- Classification report
- ROC-AUC
- Confusion matrix
- Latency benchmarking

## Notes

- The repo is production-oriented scaffolding. You should plug in real datasets under `data/raw/`.
- LSTM training expects sequentialized feature windows; if TensorFlow is missing, the project skips that model gracefully.
- Live capture requires appropriate OS-level privileges and a valid network interface.
