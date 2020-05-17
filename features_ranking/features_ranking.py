import collections

from sklearn.feature_selection import SelectKBest, chi2
from tabulate import tabulate
import pandas as pd

FeatureScore = collections.namedtuple('FeatureScore', 'name, score')


class FeaturesRanking:
    def __init__(self, features_scores=None, score_func=chi2):
        self.features_scores = features_scores
        self.selector = SelectKBest(score_func=score_func, k='all')

    def fit(self, x, y):
        self.selector.fit(x, y)
        scores = self.selector.scores_
        features_scores = [
            FeatureScore(name, round(score, 2))
            for name, score in zip(x.columns, scores)
        ]
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

    def __getitem__(self, index):
        return self.features_scores[index]
