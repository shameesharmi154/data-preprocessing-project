import os
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)


def load_data(path: str = "../data/dataset.csv") -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values:
    - Fill Age with median by Pclass and Sex when possible, else overall median
    - Fill Embarked with mode
    - Drop Cabin (too many missing)
    """
    # Age: median by Pclass and Sex
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    age_median = df.groupby(["Pclass", "Sex"])["Age"].median()

    def fill_age(row):
        if pd.isna(row["Age"]):
            try:
                med = age_median.loc[row["Pclass"], row["Sex"]]
                # If group median is NaN (no non-missing values), fall back to global median
                if pd.isna(med):
                    return df["Age"].median()
                return med
            except Exception:
                return df["Age"].median()
        return row["Age"]

    df["Age"] = df.apply(fill_age, axis=1)

    # Embarked: fill with mode
    if df["Embarked"].isna().any():
        df["Embarked"].fillna(df["Embarked"].mode().iloc[0], inplace=True)

    # Fare: convert to numeric and fill missing with median
    df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
    if df["Fare"].isna().any():
        df["Fare"].fillna(df["Fare"].median(), inplace=True)

    # Drop Cabin because too many missing values
    if "Cabin" in df.columns:
        df.drop(columns=["Cabin"], inplace=True)

    return df


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """Fix incorrect data types."""
    df["Pclass"] = df["Pclass"].astype(int)
    df["SibSp"] = df["SibSp"].astype(int)
    df["Parch"] = df["Parch"].astype(int)
    df["Survived"] = df["Survived"].astype(int)
    return df


def treat_outliers(df: pd.DataFrame, column: str = "Fare") -> pd.DataFrame:
    """Simple IQR-based winsorization for numeric column (caps outliers)."""
    if column not in df.columns:
        return df
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df


def drop_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns unlikely to be useful for preprocessing/modeling."""
    to_drop = [c for c in ["PassengerId", "Name", "Ticket"] if c in df.columns]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"WARNING: Removed {before-after} duplicate rows")
    return df


def summarize(df: pd.DataFrame) -> None:
    print("\nData summary:\n", df.info())
    print("\nMissing values:\n", df.isna().sum())


def main():
    # Load
    df = load_data(path=os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv"))
    print("Loaded dataset with shape:", df.shape)

    # Cleaning steps
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = fix_types(df)
    df = treat_outliers(df, column="Fare")
    df = drop_irrelevant(df)

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_data.csv")
    df.to_csv(out_path, index=False)
    print("cleaned_data.csv saved successfully in data/ folder")

    # Print small summary
    summarize(df)


if __name__ == "__main__":
    main()
