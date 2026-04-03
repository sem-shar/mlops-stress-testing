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

