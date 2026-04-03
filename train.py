print("Script started")

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score
import mlflow
import mlflow.sklearn
import pickle
import os

# Load data
df = pd.read_csv('data/adult.csv')

# Replace ? with NaN and drop
df = df.replace('?', pd.NA)
df = df.dropna()

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Split features and target
X = df.drop('income', axis=1)
y = df['income']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save test data for later use in tests
X_test.to_csv('data/X_test.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

# Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier()
}

os.makedirs('models', exist_ok=True)
mlflow.set_experiment("income_prediction")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        
        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"{name} - Accuracy: {accuracy:.4f}, Recall: {recall:.4f}")
        
        with open(f'models/{name}.pkl', 'wb') as f:
            pickle.dump(model, f)

# Model selection based on combined score
import json

model_scores = {}

for name in models.keys():
    with open(f'models/{name}.pkl', 'rb') as f:
        m = pickle.load(f)
    
    preds = m.predict(X_test)
    
    # Recall score
    rec = recall_score(y_test, preds)
    
    # Bias score
    X_temp = X_test.copy()
    X_temp['predictions'] = preds
    male_rate = X_temp[X_temp['sex'] == 1]['predictions'].mean()
    female_rate = X_temp[X_temp['sex'] == 0]['predictions'].mean()
    bias = abs(male_rate - female_rate)
    bias_score = 1 - bias  # higher is better
    
    # Robustness score
    X_nudged = X_test.copy()
    X_nudged['hours.per.week'] = X_nudged['hours.per.week'] + 1
    nudged_preds = m.predict(X_nudged)
    flip_rate = (preds != nudged_preds).sum() / len(preds)
    robustness_score = 1 - flip_rate  # higher is better
    
    # Combined score
    combined = (rec + bias_score + robustness_score) / 3
    model_scores[name] = combined
    print(f"{name} - Combined Score: {combined:.4f}")

best_model_name = max(model_scores, key=model_scores.get)
print(f"\nBest model: {best_model_name}")

# Save best model name for other scripts to use
with open('models/best_model.json', 'w') as f:
    json.dump({"best_model": best_model_name}, f)
