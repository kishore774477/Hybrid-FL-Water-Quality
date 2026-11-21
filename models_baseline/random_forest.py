import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import os
import joblib

def run_random_forest(X_train, y_train, X_test, y_test, seed, save_path=None):
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,          # prevents overfitting
        min_samples_split=5,
        random_state=seed
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Random Forest model saved to {save_path}")

    acc = accuracy_score(y_test, preds)
    return acc
