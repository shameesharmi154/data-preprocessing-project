"""Categorical encoding utilities
Implements:
- one_hot_encode
- label_encode
- ordinal_encode
- frequency_encode
- target_encode

Each function accepts a DataFrame and list of columns and returns a transformed DataFrame.
"""
from typing import List, Dict
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def one_hot_encode(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=cols, drop_first=False)


def label_encode(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
    return df


def ordinal_encode(df: pd.DataFrame, col_orderings: Dict[str, List[str]]) -> pd.DataFrame:
    df = df.copy()
    for col, ordering in col_orderings.items():
        enc = {val: i for i, val in enumerate(ordering)}
        df[col] = df[col].map(enc).astype(float)
    return df


def frequency_encode(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        freq = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq)
    return df


def target_encode(df: pd.DataFrame, cols: List[str], target: str, smoothing: float = 1.0) -> pd.DataFrame:
    """Target encoding with smoothing.
    smoothing: higher = stronger regularization toward global mean
    """
    df = df.copy()
    global_mean = df[target].mean()
    for col in cols:
        agg = df.groupby(col)[target].agg(["mean", "count"])
        smoothing_factor = 1 / (1 + np.exp(-(agg["count"] - smoothing)))
        agg["smoothed"] = global_mean * (1 - smoothing_factor) + agg["mean"] * smoothing_factor
        mapping = agg["smoothed"].to_dict()
        df[col + "_te"] = df[col].map(mapping).fillna(global_mean)
    return df


def demo():
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    df = pd.read_csv(os.path.join(base, "cleaned_data.csv"))

    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    print("Categorical columns found:", cat_cols)

    if "Survived" in df.columns:
        # target encoding demo uses Survived
        te = target_encode(df, ["Sex"], target="Survived", smoothing=1.0)
        te.to_csv(os.path.join(base, "encoded_target.csv"), index=False)
        print("target-encoded file saved to data/encoded_target.csv")

    ohe = one_hot_encode(df, [c for c in cat_cols if c != "Survived"] )
    ohe.to_csv(os.path.join(base, "encoded_ohe.csv"), index=False)
    print("one-hot-encoded file saved to data/encoded_ohe.csv")

    le = label_encode(df, cat_cols)
    le.to_csv(os.path.join(base, "encoded_label.csv"), index=False)
    print("label-encoded file saved to data/encoded_label.csv")


if __name__ == "__main__":
    demo()
