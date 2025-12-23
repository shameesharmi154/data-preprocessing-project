import pandas as pd
from src import categorical_encoding as ce


def test_one_hot_encode_creates_columns():
    df = pd.DataFrame({'Sex':['male','female'], 'Embarked':['S','C']})
    ohe = ce.one_hot_encode(df, ['Sex','Embarked'])
    # Expect columns like Sex_male and Embarked_S
    assert any(col.startswith('Sex_') for col in ohe.columns)
    assert any(col.startswith('Embarked_') for col in ohe.columns)


def test_target_encode_creates_te_column():
    df = pd.DataFrame({'Sex':['male','female','male'], 'Survived':[1,0,1]})
    te = ce.target_encode(df, ['Sex'], target='Survived')
    assert 'Sex_te' in te.columns
    assert te['Sex_te'].isna().sum() == 0
