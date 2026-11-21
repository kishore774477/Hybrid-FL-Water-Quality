import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import os
import joblib

def run_logistic_regression(X_train, y_train, X_test, y_test, save_path=None):
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Logistic Regression model saved to {save_path}")

    acc = accuracy_score(y_test, preds)
    return acc
