import numpy as np
import pandas as pd

def simulate_accelerometer_data(n_records=1000):
    np.random.seed(42)
    time = np.linspace(0, 10, n_records)
    
    # Normal motor devir frekansı (~50 Hz temel bileşen)
    base_freq = 50.0
    normal_signal = 1.2 * np.sin(2 * np.pi * base_freq * time) + np.random.normal(0, 0.2, n_records)
    
    # Rulman/bilye hasarlı motor: periyodik darbe ve yüksek harmonikler
    fault_signal = (
        1.1 * np.sin(2 * np.pi * base_freq * time) 
        + 0.8 * np.sin(2 * np.pi * 3 * base_freq * time)
        + np.random.normal(0, 0.45, n_records)
    )
    # Ani mekanik vuruntular
    shock_points = np.random.choice(n_records, size=30, replace=False)
    fault_signal[shock_points] += np.random.choice([-1.5, 1.5], size=30)
    
    df_normal = pd.DataFrame({"time": time, "vibration": normal_signal, "label": 0})
    df_fault = pd.DataFrame({"time": time, "vibration": fault_signal, "label": 1})
    
    dataset = pd.concat([df_normal, df_fault], ignore_index=True)
    dataset.to_csv("vibration_raw.csv", index=False)
    print("vibration_raw.csv generated successfully.")

if __name__ == "__main__":
    simulate_accelerometer_data()
