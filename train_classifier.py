import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def extract_features(signal_window):
    """Zaman domeni istatistiksel öznitelikleri"""
    rms = np.sqrt(np.mean(signal_window**2))
    std = np.std(signal_window)
    peak = np.max(np.abs(signal_window))
    kurt = kurtosis(signal_window)
    skw = skew(signal_window)
    crest_factor = peak / (rms + 1e-6)
    
    return [rms, std, peak, kurt, skw, crest_factor]

def build_dataset_from_windows(df, window_size=50):
    X, y = [], []
    for label in [0, 1]:
        subset = df[df['label'] == label]['vibration'].values
        n_windows = len(subset) // window_size
        for i in range(n_windows):
            window = subset[i*window_size : (i+1)*window_size]
            features = extract_features(window)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

def main():
    try:
        df = pd.read_csv("vibration_raw.csv")
    except FileNotFoundError:
        print("Data not found. Running generate_data first...")
        from generate_data import simulate_accelerometer_data
        simulate_accelerometer_data()
        df = pd.read_csv("vibration_raw.csv")

    X, y = build_dataset_from_windows(df, window_size=50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("\n--- Test Set Results ---")
    print(classification_report(y_test, preds, target_names=["Nominal", "Faulty"]))

if __name__ == "__main__":
    main()
