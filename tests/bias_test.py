print("bias test started")
import pandas as pd
import pickle
import sys
import json


with open('models/best_model.json', 'r') as f:
    best_model_name = json.load(f)['best_model']

with open(f'models/{best_model_name}.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"Testing model: {best_model_name}")

# Load test data
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv').squeeze()

predictions = model.predict(X_test)
X_test = X_test.copy()
X_test['predictions'] = predictions

# sex column: 1 = male, 0 = female (after LabelEncoder in train.py)
male = X_test[X_test['sex'] == 1]['predictions'].mean()
female = X_test[X_test['sex'] == 0]['predictions'].mean()

demographic_parity = abs(male - female)
print(f"Male positive rate:   {male:.4f}")
print(f"Female positive rate: {female:.4f}")
print(f"Demographic parity difference: {demographic_parity:.4f}")

# Threshold of 0.2 corresponds to the EEOC four-fifths rule:
# a selection rate below 80% of the highest group's rate signals
# adverse impact. In practice this maps to roughly a 0.2 gap
# in positive prediction rates between demographic groups.
THRESHOLD = 0.2
if demographic_parity <= THRESHOLD:
    print("BIAS TEST: PASSED")
else:
    print("BIAS TEST: FAILED")
    sys.exit(1)

