"""Example script to run the full pipeline end-to-end.
Usage: python examples/run_pipeline.py
"""
import os
import sys

# Add project root to path so src can be imported when running from examples/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src import data_cleaning, categorical_encoding, feature_scaling, model_train


def run_all():
    print("1) Running data cleaning...")
    data_cleaning.main()

    print("2) Running categorical encodings (OHE & Target demo)...")
    categorical_encoding.demo()

    print("3) Running feature scaling demos...")
    feature_scaling.demo()

    print("4) Training demo model using OHE data...")
    import pandas as pd
    encoded_path = os.path.join(os.path.dirname(__file__), "..", "data", "encoded_ohe.csv")
    df = pd.read_csv(encoded_path)
    model_train.train_and_evaluate(df)

    print("All steps finished. Check `data/` and `reports/` for outputs.")


if __name__ == "__main__":
    run_all()