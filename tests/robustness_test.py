import pandas as pd
import pickle
import sys
import json
import numpy as np

print("robustness test started")


with open('models/best_model.json', 'r') as f:
    best_model_name = json.load(f)['best_model']

with open(f'models/{best_model_name}.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"Testing model: {best_model_name}")

# Load test data
X_test = pd.read_csv('data/X_test.csv')

# Original predictions on clean data — this is our baseline
original_predictions = model.predict(X_test)


features_to_perturb = ['hours.per.week', 'capital.gain', 'age']

# Noise levels: 10%, 50%, 100% of each feature's std
noise_levels = [0.1, 0.5]

print("\n--- Progressive Perturbation Results ---")
all_flip_rates = []

for feature in features_to_perturb:
    feature_std = X_test[feature].std()
    print(f"\nFeature: {feature} (std={feature_std:.2f})")

    for noise_scale in noise_levels:
        sigma = noise_scale * feature_std

        # Add Gaussian noise scaled to feature std
        X_perturbed = X_test.copy()
        noise = np.random.normal(0, sigma, len(X_test))
        X_perturbed[feature] = X_perturbed[feature] + noise

        perturbed_predictions = model.predict(X_perturbed)
        flips = (original_predictions != perturbed_predictions).sum()
        flip_rate = flips / len(original_predictions)
        all_flip_rates.append(flip_rate)

        print(f"  Noise = {noise_scale:.0%} of std (sigma={sigma:.2f}) "
              f"-> Flip rate: {flip_rate:.4f} ({flips}/{len(original_predictions)})")


max_flip_rate = max(all_flip_rates)
THRESHOLD = 0.15

print(f"\nMax flip rate across all perturbations: {max_flip_rate:.4f}")
print(f"Threshold: {THRESHOLD}")

if max_flip_rate <= THRESHOLD:
    print("ROBUSTNESS TEST: PASSED")
else:
    print("ROBUSTNESS TEST: FAILED")
    sys.exit(1)
