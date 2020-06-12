import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix as sk_plot_confusion_matrix

from constants import PLOTS_DIR


def plot_params_scores(results):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    _, ax = plt.subplots()
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=results,
        x='param_selector__k',
        y='mean_test_score',
        hue='params_key',
        legend=False
    )
    plt.ylim(0.7, 1)
    plt.xlabel('Attributes number')
    plt.ylabel('f1-score')
    plt.legend(title='Parameters', loc='upper right', labels=results['params_key'].unique())
    plt.savefig(os.path.join(PLOTS_DIR, 'scores.png'))


def plot_confusion_matrix(estimator, x, y_true):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    sk_plot_confusion_matrix(estimator, x, y_true, values_format='d', cmap=plt.cm.Blues)
    plt.savefig(os.path.join(PLOTS_DIR, 'cm.png'))
