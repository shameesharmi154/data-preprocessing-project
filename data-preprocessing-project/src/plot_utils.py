"""Reusable plotting utilities used by the notebook and tests."""
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_age_histogram(df, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(df["Age"].dropna(), bins=20, color="#2b7cff", alpha=0.8)
    ax.set_title("Age distribution")
    ax.set_xlabel("Age")
    if save_path:
        fig.savefig(save_path, format="svg")
    return fig, ax
