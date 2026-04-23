# AI-Based Network Traffic Anomaly Detection System

An end-to-end network anomaly detection platform that uses machine learning and deep learning to identify suspicious traffic patterns, monitor congestion, and visualize insights through an interactive dashboard.

## Overview

This project is designed to help analyze network traffic data and detect anomalous behavior that may indicate attacks, misuse, or unusual activity. It combines data ingestion, preprocessing, feature engineering, multiple anomaly-detection models, and a Streamlit dashboard for real-time monitoring and reporting.

## Key Features

- End-to-end pipeline for loading, preprocessing, training, and inference
- Support for CICIDS-style network traffic datasets
- Multiple detection models including `IsolationForest`, LSTM autoencoder, and gradient-boosted classification
- Weighted ensemble scoring for stronger anomaly detection
- Real-time alerting and congestion monitoring
- Interactive Streamlit dashboard for monitoring and analytics
- Basic automated tests for core detection logic

## Tech Stack

- Python
- Scikit-learn
- TensorFlow/Keras
- XGBoost
- Pandas and NumPy
- Streamlit
- Pytest

## Project Structure

```text
AI-Based-Network-Traffic-Anomaly-Detection-System/
+-- data/
|   +-- raw/
|   +-- processed/
+-- dashboard/
|   +-- app.py
|   +-- pages/
+-- docs/
+-- notebooks/
+-- src/
|   +-- models/
|   +-- alerting.py
|   +-- cli.py
|   +-- config.py
|   +-- data_loader.py
|   +-- detector.py
|   +-- feature_extractor.py
|   +-- pipeline.py
|   +-- preprocessing.py
+-- tests/
+-- requirements.txt
+-- README.md
```

## Quick Start

1. Create and activate a virtual environment.
2. Install the project dependencies.
3. Run the training and detection pipeline.
4. Launch the dashboard for visual monitoring.

```bash
pip install -r requirements.txt
python -m src.cli run-all --data-path data/raw/your_dataset.csv
streamlit run dashboard/app.py
```

## Pipeline Phases

### 1. Data Collection

- Load CICIDS, UNSW, or similar network traffic datasets
- Support synthetic sample generation for local testing
- Optional hooks for live packet capture integrations

### 2. Preprocessing and Feature Engineering

- Clean and normalize raw traffic data
- Handle missing values and categorical labels
- Scale features for downstream models
- Optional class balancing using SMOTE

### 3. Model Training

- `IsolationForest` for unsupervised anomaly scoring
- LSTM autoencoder for reconstruction-based anomaly detection
- XGBoost classifier with safe fallback behavior when unavailable
- Ensemble fusion for improved detection performance

### 4. Real-Time Detection

- Sliding-window anomaly scoring
- Traffic congestion estimation
- Severity-based alert generation

### 5. Monitoring Dashboard

- Live metrics and anomaly trends
- Flow analysis and visualization
- Alert inspection and execution tracking
- Reporting-focused admin views

### 6. Testing and Evaluation

- Classification metrics
- ROC-AUC analysis
- Confusion matrix evaluation
- Latency benchmarking

## Use Cases

- Intrusion detection research
- Security analytics demonstrations
- Network monitoring prototypes
- AI-based cyber threat detection projects

## Notes

- Large raw datasets and generated outputs are excluded from version control.
- Place your datasets inside `data/raw/` before running the pipeline.
- If TensorFlow is unavailable, the LSTM-based model may be skipped gracefully.
- Live capture features may require additional OS-level privileges and supported interfaces.

## GitHub Description

AI-powered network traffic anomaly detection system using machine learning, deep learning, and a Streamlit dashboard for real-time monitoring.

## Suggested GitHub Topics

`cybersecurity`, `network-security`, `anomaly-detection`, `machine-learning`, `deep-learning`, `streamlit`, `python`, `intrusion-detection`, `xgboost`, `lstm`