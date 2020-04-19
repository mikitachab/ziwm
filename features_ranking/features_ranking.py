import collections

from sklearn.feature_selection import SelectKBest, chi2
from tabulate import tabulate
import pandas as pd

FeatureScore = collections.namedtuple('FeatureScore', 'name, score')


def make_features_ranking(x, y):
    k_best_selector = SelectKBest(score_func=chi2, k='all')
    k_best_selector.fit(x, y)
    scores = k_best_selector.scores_
    features_scores = [
        FeatureScore(name, score)
        for name, score in zip(x.columns, scores)
    ]
    return FeaturesRanking(features_scores)


class FeaturesRanking:
    def __init__(self, features_scores):
        self.features_scores = sorted(features_scores, key=lambda s: s.score, reverse=True)

    def to_dataframe(self):
        df = {
            'name': [s.name for s in self.features_scores],
            'score': [s.score for s in self.features_scores]
        }
        return pd.DataFrame(df)

    def to_latex(self):
        return self.to_dataframe().to_latex()

    def __str__(self):
        df = self.to_dataframe()
        return tabulate(df, headers=df.columns)
