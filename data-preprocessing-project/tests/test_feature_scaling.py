import pandas as pd
from src.feature_scaling import min_max_scale


def test_min_max_scaling_range():
    df = pd.DataFrame({'A':[0,5,10], 'B':[1,2,3]})
    out = min_max_scale(df, ['A','B'])
    assert out['A'].min() >= 0
    assert out['A'].max() <= 1
    assert out['B'].min() >= 0
    assert out['B'].max() <= 1
