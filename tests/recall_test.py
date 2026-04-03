import pandas as pd
import pickle
import sys
from sklearn.metrics import recall_score

import json

with open('models/best_model.json', 'r') as f:
    best_model_name = json.load(f)['best_model']

with open(f'models/{best_model_name}.pkl', 'rb') as f:
    model = pickle.load(f)

print("recall test started")

X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv').squeeze()

with open('models/RandomForest.pkl', 'rb') as f:
    model = pickle.load(f)

predictions = model.predict(X_test)
recall = recall_score(y_test, predictions)

print(f"Recall: {recall:.4f}")

THRESHOLD = 0.5
if recall >= THRESHOLD:
    print("RECALL TEST: PASSED")
else:
    print("RECALL TEST: FAILED")
    sys.exit(1)