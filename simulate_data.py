import pandas as pd
import numpy as np
import requests
import json

# Load original test data as base
X_test = pd.read_csv('data/X_test.csv')

# Simulate drift - increase hours per week distribution
X_simulated = X_test.copy()
X_simulated['hours.per.week'] = X_simulated['hours.per.week'] * 1.2
X_simulated['age'] = X_simulated['age'] + np.random.normal(0, 2, len(X_simulated))

# Save simulated data for drift detection
X_simulated.to_csv('data/simulated_data.csv', index=False)
print(f"Simulated {len(X_simulated)} records")

# Send to API
url = "http://127.0.0.1:8000/predict"
success = 0
failed = 0

for _, row in X_simulated.head(100).iterrows():
    payload = {
        "age": row['age'],
        "workclass": row['workclass'],
        "fnlwgt": row['fnlwgt'],
        "education": row['education'],
        "education_num": row['education.num'],
        "marital_status": row['marital.status'],
        "occupation": row['occupation'],
        "relationship": row['relationship'],
        "race": row['race'],
        "sex": row['sex'],
        "capital_gain": row['capital.gain'],
        "capital_loss": row['capital.loss'],
        "hours_per_week": row['hours.per.week'],
        "native_country": row['native.country']
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            success += 1
        else:
            failed += 1
    except:
        failed += 1

print(f"Successful predictions: {success}")
print(f"Failed predictions: {failed}")