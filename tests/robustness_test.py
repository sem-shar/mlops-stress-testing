import pandas as pd
import pickle
import sys

print("robustness test started")

# Load test data
X_test = pd.read_csv('data/X_test.csv')

# Load best model
with open('models/RandomForest.pkl', 'rb') as f:
    model = pickle.load(f)

# Original predictions
original_predictions = model.predict(X_test)

# Nudge hours per week by 1
X_test_nudged = X_test.copy()
X_test_nudged['hours.per.week'] = X_test_nudged['hours.per.week'] + 1

# Nudged predictions
nudged_predictions = model.predict(X_test_nudged)

# Calculate flip rate
flips = (original_predictions != nudged_predictions).sum()
flip_rate = flips / len(original_predictions)

print(f"Total predictions: {len(original_predictions)}")
print(f"Flipped predictions: {flips}")
print(f"Flip rate: {flip_rate:.4f}")

THRESHOLD = 0.15
if flip_rate <= THRESHOLD:
    print("ROBUSTNESS TEST: PASSED")
else:
    print("ROBUSTNESS TEST: FAILED")
    sys.exit(1)
