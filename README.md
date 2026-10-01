# UAV Engine Vibration Analysis & Fault Classification

Lightweight pipeline for detecting mechanical anomalies (bearing/motor degradation) in UAV telemetry using time-domain vibration metrics.

## Methodology
Instead of feeding raw signal arrays directly into heavy networks, the pipeline computes statistical moments over rolling windows:
- **RMS & Peak Amplitude:** Energy level indicators
- **Kurtosis & Skewness:** Distribution shape metrics sensitive to mechanical impacts
- **Crest Factor:** Peak-to-RMS ratio for impulsive shock detection

Features are classified using a constrained Random Forest baseline.

## Quickstart

```bash
pip install -r requirements.txt
python generate_data.py
python train_classifier.py
