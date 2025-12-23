import pandas as pd
import numpy as np
from src import data_cleaning


def test_handle_missing_values_fills_age_and_embarked():
    df = pd.DataFrame({
        'Pclass': [1, 3],
        'Sex': ['male', 'female'],
        'Age': [np.nan, 10],
        'Fare': [np.nan, 20],
        'Embarked': [None, 'S']
    })
    out = data_cleaning.handle_missing_values(df.copy())
    assert out['Age'].isna().sum() == 0
    assert out['Embarked'].isna().sum() == 0


def test_drop_irrelevant():
    df = pd.DataFrame({'PassengerId':[1], 'Name':['a'], 'Ticket':['t'], 'Age':[20]})
    out = data_cleaning.drop_irrelevant(df.copy())
    assert 'PassengerId' not in out.columns
    assert 'Name' not in out.columns
    assert 'Ticket' not in out.columns
