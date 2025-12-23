"""Simple model training demo:
- Loads cleaned and encoded data
- Performs train/test split
- Trains a RandomForestClassifier
- Saves basic metrics to reports/metrics.txt
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


def load_data(path: str = "../data/encoded_ohe.csv"):
    return pd.read_csv(path)


def train_and_evaluate(df, target: str = "Survived"):
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "reports"), exist_ok=True)

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42, n_estimators=100)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)

    metrics = f"accuracy: {acc:.4f}\nprecision: {prec:.4f}\nrecall: {rec:.4f}\n"
    with open(os.path.join(os.path.dirname(__file__), "..", "reports", "metrics.txt"), "w") as f:
        f.write(metrics)

    print("Model trained and metrics saved to reports/metrics.txt")
    print(metrics)


if __name__ == "__main__":
    df = load_data(os.path.join(os.path.dirname(__file__), "..", "data", "encoded_ohe.csv"))
    if "Survived" not in df.columns:
        raise RuntimeError("encoded_ohe.csv must include target column 'Survived' for training demo")
    train_and_evaluate(df)
