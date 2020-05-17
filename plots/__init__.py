import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from constants import PLOTS_DIR


def plot_params_scores(scores):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    for score in scores:
        plot_score(score)


def plot_score(score):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=pd.DataFrame(dict(
            k=score.k,
            scores=score.scores
        )),
        x='k',
        y='scores'
    )
    title = f'metric: {score.metric} n_neighbors: {score.n_neighbors}'
    plt.title(title)
    plt.ylabel('accuracy')
    plt.savefig(make_plot_filename(score))


def make_plot_filename(score):
    plot_title = f'metric_{score.metric}_n_neighbors:_{score.n_neighbors}.png'
    return os.path.join(PLOTS_DIR, plot_title)
