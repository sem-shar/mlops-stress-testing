print("bias test started")
import pandas as pd
import pickle
import sys

# Load test data
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv').squeeze()

# Load best model
with open('models/RandomForest.pkl', 'rb') as f:
    model = pickle.load(f)

predictions = model.predict(X_test)
X_test = X_test.copy()
X_test['predictions'] = predictions

# sex column - 1 is male, 0 is female (after label encoding)
male = X_test[X_test['sex'] == 1]['predictions'].mean()
female = X_test[X_test['sex'] == 0]['predictions'].mean()

demographic_parity = abs(male - female)
print(f"Male positive rate: {male:.4f}")
print(f"Female positive rate: {female:.4f}")
print(f"Demographic parity difference: {demographic_parity:.4f}")

THRESHOLD = 0.2
if demographic_parity <= THRESHOLD:
    print("BIAS TEST: PASSED")
else:
    print("BIAS TEST: FAILED")
    sys.exit(1)
