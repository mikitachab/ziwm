import pandas as pd

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from .experiment_params import get_experiment_params
from .constants import CV_RANDOM_STATE, PREDEFINED_N_NEIGHBORS, PREDEFINED_METRICS

from features_ranking import make_features_ranking


def prepare_data(dataset, order_by_feature_score=True):
    data = dataset.iloc[:, 0:len(dataset.columns) - 1]
    target = dataset.iloc[:, len(dataset.columns) - 1]

    if order_by_feature_score:
        columns_ordered_by_feat_score = make_features_ranking(data, target).to_dataframe()['name'].to_list()
        data = data[columns_ordered_by_feat_score]
    return data, target


def prepare_pipeline(model, feat_selector):
    return Pipeline(
        [
            ('selector', feat_selector),
            ('model', model)
        ]
    )


def prepare_experiment(dataset, predefined_params=True):
    data, target = prepare_data(dataset)
    ex_params = get_experiment_params(data)
    n_splits, n_repeats = ex_params.get_cv_params()

    pipe = prepare_pipeline(KNeighborsClassifier(), SelectKBest(score_func=chi2))

    param_grid = ex_params.get_param_grid()
    if predefined_params:
        param_grid['model__n_neighbors'] = PREDEFINED_N_NEIGHBORS
        param_grid['model__metric'] = PREDEFINED_METRICS

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=CV_RANDOM_STATE)

    return Experiment(pipe, data, target, param_grid, cv)


class Experiment:
    def __init__(self, pipe, data, target, params, cv):
        self.data = data
        self.attributes = data.columns.to_list()
        self.target = target
        self.grid_search = GridSearchCV(pipe, param_grid=params, cv=cv)

    def run_experiment(self):
        self.grid_search.fit(self.data, self.target)

    def get_results(self):
        return pd.DataFrame(self.grid_search.cv_results_)
