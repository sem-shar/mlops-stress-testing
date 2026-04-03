import pandas as pd
import numpy as np
import sys

print("drift test started")

# Load original and simulated data
original = pd.read_csv('data/X_test.csv')
simulated = pd.read_csv('data/simulated_data.csv')

def calculate_psi(original, simulated, bins=10):
    min_val = min(original.min(), simulated.min())
    max_val = max(original.max(), simulated.max())
    
    orig_hist, edges = np.histogram(original, bins=bins, range=(min_val, max_val))
    sim_hist, _ = np.histogram(simulated, bins=bins, range=(min_val, max_val))
    
    orig_hist = orig_hist / len(original)
    sim_hist = sim_hist / len(simulated)
    
    orig_hist = np.where(orig_hist == 0, 0.0001, orig_hist)
    sim_hist = np.where(sim_hist == 0, 0.0001, sim_hist)
    
    psi = np.sum((sim_hist - orig_hist) * np.log(sim_hist / orig_hist))
    return psi

# Check drift on key features
features = ['hours.per.week', 'age', 'capital.gain']
drift_detected = False

for feature in features:
    psi = calculate_psi(original[feature], simulated[feature])
    status = "DRIFT DETECTED" if psi > 0.1 else "NO DRIFT"
    print(f"{feature} - PSI: {psi:.4f} - {status}")
    if psi > 0.25:
        drift_detected = True

if drift_detected:
    print("DRIFT TEST: SIGNIFICANT DRIFT DETECTED - RETRAINING NEEDED")
    sys.exit(1)
else:
    print("DRIFT TEST: PASSED")