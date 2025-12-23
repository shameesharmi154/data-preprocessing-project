"""Feature scaling utilities demonstrating:
- Min-Max Scaling
- MaxAbs Scaling
- Vector Normalization (L2)
- Standardization (Z-score)
"""
import os
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, Normalizer, StandardScaler


def get_numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()


def min_max_scale(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def max_abs_scale(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    scaler = MaxAbsScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def l2_normalize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    scaler = Normalizer(norm="l2")
    df[cols] = scaler.fit_transform(df[cols])
    return df


def standard_scale(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def demo():
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    df = pd.read_csv(os.path.join(base, "cleaned_data.csv"))
    num_cols = get_numeric_cols(df)
    print("Numeric columns:", num_cols)

    minmax = min_max_scale(df, num_cols)
    minmax.to_csv(os.path.join(base, "scaled_minmax.csv"), index=False)
    print("Min-Max scaled data saved to data/scaled_minmax.csv")

    maxabs = max_abs_scale(df, num_cols)
    maxabs.to_csv(os.path.join(base, "scaled_maxabs.csv"), index=False)
    print("MaxAbs scaled data saved to data/scaled_maxabs.csv")

    l2 = l2_normalize(df, num_cols)
    l2.to_csv(os.path.join(base, "scaled_l2.csv"), index=False)
    print("L2-normalized data saved to data/scaled_l2.csv")

    std = standard_scale(df, num_cols)
    std.to_csv(os.path.join(base, "scaled_standard.csv"), index=False)
    print("Standard scaled data saved to data/scaled_standard.csv")


if __name__ == "__main__":
    demo()
