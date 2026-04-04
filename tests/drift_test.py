import pandas as pd
import numpy as np
import pickle
import json
import sys

print("drift test started")

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

features = ['hours.per.week', 'age', 'capital.gain']
input_drift_detected = False

print("\n--- Layer 1: Input Distribution Drift (PSI) ---")
for feature in features:
    psi = calculate_psi(original[feature], simulated[feature])
    status = "DRIFT DETECTED" if psi > 0.1 else "NO DRIFT"
    print(f"{feature} - PSI: {psi:.4f} - {status}")
    if psi > 0.25:
        input_drift_detected = True


print("\n--- Layer 2: Behavioral Drift (Prediction Rate) ---")

behavioral_drift_detected = False

# Compute baseline approval rate from saved model on clean data
with open('models/best_model.json', 'r') as f:
    best_model_name = json.load(f)['best_model']

with open(f'models/{best_model_name}.pkl', 'rb') as f:
    model = pickle.load(f)

baseline_predictions = model.predict(original)
baseline_approval_rate = baseline_predictions.mean()
print(f"Baseline approval rate (clean data, {best_model_name}): {baseline_approval_rate:.4f}")

# Load simulated approval rate from simulate_data.py output
try:
    with open('data/drift_summary.json', 'r') as f:
        drift_summary = json.load(f)

    if drift_summary['api_available'] and drift_summary['simulated_approval_rate'] is not None:
        # Full mode: API was running, we have real prediction distribution
        simulated_rate = drift_summary['simulated_approval_rate']
        print(f"Simulated approval rate (from API, n={drift_summary['sample_size']}): {simulated_rate:.4f}")

        approval_rate_shift = abs(simulated_rate - baseline_approval_rate)
        print(f"Approval rate shift: {approval_rate_shift:.4f}")

        # A shift of more than 5 percentage points in approval rate
        # indicates the model is behaving meaningfully differently
        # on shifted data — retraining signal
        BEHAVIORAL_THRESHOLD = 0.05
        if approval_rate_shift > BEHAVIORAL_THRESHOLD:
            behavioral_drift_detected = True
            print(f"Behavioral drift detected (shift={approval_rate_shift:.4f} > {BEHAVIORAL_THRESHOLD})")
        else:
            print(f"No significant behavioral drift (shift={approval_rate_shift:.4f})")

    else:
        # Fallback mode: API not available (CI without running server)
       
        print("API not available — computing behavioral drift directly from model")
        sim_predictions = model.predict(simulated)
        simulated_rate = sim_predictions.mean()
        approval_rate_shift = abs(simulated_rate - baseline_approval_rate)

        print(f"Simulated approval rate (direct model): {simulated_rate:.4f}")
        print(f"Approval rate shift: {approval_rate_shift:.4f}")

        BEHAVIORAL_THRESHOLD = 0.05
        if approval_rate_shift > BEHAVIORAL_THRESHOLD:
            behavioral_drift_detected = True
            print(f"Behavioral drift detected (shift={approval_rate_shift:.4f} > {BEHAVIORAL_THRESHOLD})")
        else:
            print(f"No significant behavioral drift (shift={approval_rate_shift:.4f})")

except FileNotFoundError:
    print("drift_summary.json not found — simulate_data.py may not have run")
    print("Skipping behavioral drift check")

# --- Final Gate ---
print("\n--- Drift Test Summary ---")
print(f"Input drift (PSI > 0.25 on any feature): {input_drift_detected}")
print(f"Behavioral drift (approval rate shift > 5%): {behavioral_drift_detected}")

if input_drift_detected or behavioral_drift_detected:
    print("DRIFT TEST: SIGNIFICANT DRIFT DETECTED - RETRAINING NEEDED")
    sys.exit(1)
else:
    print("DRIFT TEST: PASSED")
