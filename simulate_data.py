import pandas as pd
import numpy as np
import requests
import json



# Load original test data as base
X_test = pd.read_csv('data/X_test.csv')

# --- Simulate covariate shift ---

X_simulated = X_test.copy()
X_simulated['hours.per.week'] = X_simulated['hours.per.week'] * 1.2
X_simulated['age'] = X_simulated['age'] + np.random.normal(0, 2, len(X_simulated))

# Save shifted feature data for PSI calculation in drift_test.py
X_simulated.to_csv('data/simulated_data.csv', index=False)
print(f"Simulated {len(X_simulated)} records with covariate shift")



API_URL = "http://localhost:8000/predict"

# Map from CSV column names (dot) to API field names (underscore)
column_map = {
    'age': 'age',
    'workclass': 'workclass',
    'fnlwgt': 'fnlwgt',
    'education': 'education',
    'education.num': 'education_num',
    'marital.status': 'marital_status',
    'occupation': 'occupation',
    'relationship': 'relationship',
    'race': 'race',
    'sex': 'sex',
    'capital.gain': 'capital_gain',
    'capital.loss': 'capital_loss',
    'hours.per.week': 'hours_per_week',
    'native.country': 'native_country'
}

X_api = X_simulated.rename(columns=column_map)

# Collect predictions from API

SAMPLE_SIZE = min(200, len(X_api))
X_sample = X_api.sample(n=SAMPLE_SIZE, random_state=42)

simulated_predictions = []
api_available = True

try:
    for _, row in X_sample.iterrows():
        payload = row.to_dict()
        response = requests.post(API_URL, json=payload, timeout=3)
        if response.status_code == 200:
            simulated_predictions.append(response.json()['prediction'])
        else:
            print(f"API returned status {response.status_code}")
            api_available = False
            break
except requests.exceptions.ConnectionError:
    # API not running in CI — this is expected during pipeline
    # The drift test will fall back to PSI-only mode
    api_available = False
    print("API not reachable — skipping prediction distribution capture")
    print("(Run app.py locally to enable full behavioral drift monitoring)")

if api_available and simulated_predictions:
    simulated_approval_rate = sum(simulated_predictions) / len(simulated_predictions)
    print(f"Simulated approval rate (from API): {simulated_approval_rate:.4f}")

    # Save for drift_test.py to compare against baseline
    drift_summary = {
        "simulated_approval_rate": simulated_approval_rate,
        "sample_size": len(simulated_predictions),
        "api_available": True
    }
else:
    drift_summary = {
        "simulated_approval_rate": None,
        "sample_size": 0,
        "api_available": False
    }

with open('data/drift_summary.json', 'w') as f:
    json.dump(drift_summary, f, indent=2)

print("Simulation complete. drift_summary.json saved.")
