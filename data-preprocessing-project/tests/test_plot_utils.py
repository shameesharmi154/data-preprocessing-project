import os
import pandas as pd
import matplotlib
from src.plot_utils import plot_age_histogram


def test_plot_age_histogram_returns_fig_ax(tmp_path):
    df = pd.DataFrame({"Age": [10, 20, 30, 40, None]})
    fig, ax = plot_age_histogram(df, save_path=str(tmp_path / "age.svg"))
    assert hasattr(fig, 'savefig')
    assert ax.get_title() == 'Age distribution'
    assert ax.get_xlabel() == 'Age'
    # saved file exists
    assert (tmp_path / "age.svg").exists()
