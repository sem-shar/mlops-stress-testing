import pandas as pd
import pickle
import sys
import json
import numpy as np
from sklearn.metrics import recall_score

print("recall test started")

# FIX 1: Use best model from selection — not hardcoded RandomForest
with open('models/best_model.json', 'r') as f:
    best_model_name = json.load(f)['best_model']

with open(f'models/{best_model_name}.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"Testing model: {best_model_name}")

# Load clean test data and labels
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv').squeeze()


baseline_predictions = model.predict(X_test)
baseline_recall = recall_score(y_test, baseline_predictions)
print(f"\nBaseline recall (clean data): {baseline_recall:.4f}")

BASELINE_THRESHOLD = 0.5
if baseline_recall < BASELINE_THRESHOLD:
    print(f"RECALL TEST: FAILED on clean data (recall={baseline_recall:.4f} < {BASELINE_THRESHOLD})")
    sys.exit(1)

print("\n--- Recall Under Simulated Drift Scenarios ---")

recall_drops = []

# Scenario 1: Income distribution shift

X_drift_1 = X_test.copy()
X_drift_1['capital.gain'] = X_drift_1['capital.gain'] * 1.5
preds_1 = model.predict(X_drift_1)
recall_1 = recall_score(y_test, preds_1)
drop_1 = baseline_recall - recall_1
recall_drops.append(drop_1)
print(f"Scenario 1 (capital gain +50%):     recall={recall_1:.4f}  drop={drop_1:+.4f}")

# Scenario 2: Working hours distribution shift

X_drift_2 = X_test.copy()
X_drift_2['hours.per.week'] = X_drift_2['hours.per.week'] * 1.2
preds_2 = model.predict(X_drift_2)
recall_2 = recall_score(y_test, preds_2)
drop_2 = baseline_recall - recall_2
recall_drops.append(drop_2)
print(f"Scenario 2 (hours/week +20%):       recall={recall_2:.4f}  drop={drop_2:+.4f}")

# Scenario 3: Age distribution shift

X_drift_3 = X_test.copy()
X_drift_3['age'] = X_drift_3['age'] + np.random.normal(5, 2, len(X_test))
preds_3 = model.predict(X_drift_3)
recall_3 = recall_score(y_test, preds_3)
drop_3 = baseline_recall - recall_3
recall_drops.append(drop_3)
print(f"Scenario 3 (age shift +5 years):    recall={recall_3:.4f}  drop={drop_3:+.4f}")


# A drop of 0.10 means recall fell by 10 percentage points under
# drift. 
max_drop = max(recall_drops)
DEGRADATION_THRESHOLD = 0.10

print(f"\nMax recall degradation under drift: {max_drop:+.4f}")
print(f"Degradation threshold: {DEGRADATION_THRESHOLD}")

if max_drop <= DEGRADATION_THRESHOLD:
    print("RECALL STRESS TEST: PASSED")
else:
    print("RECALL STRESS TEST: FAILED — recall degrades too much under distribution shift")
    sys.exit(1)
