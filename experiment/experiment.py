from collections import namedtuple

import pandas as pd

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from .experiment_params import get_experiment_params
from .constants import CV_RANDOM_STATE, PREDEFINED_N_NEIGHBORS, PREDEFINED_METRICS
import plots

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


ExperimentResult = namedtuple('ExperimentResult', 'results, best_score, best_params')


class Experiment:
    def __init__(self, pipe, data, target, params, cv, scoring='f1'):
        self.data = data
        self.attributes = data.columns.to_list()
        self.target = target
        self.grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=params,
            cv=cv,
            scoring=scoring
        )

    def run_experiment(self, make_plots=False):
        self.grid_search.fit(self.data, self.target)
        cv_results = pd.DataFrame(self.grid_search.cv_results_)
        cv_results['params_key'] = cv_results['params'].apply(_make_params_key)
        self.result = ExperimentResult(
            results=cv_results,
            best_params=self.grid_search.best_params_,
            best_score=self.grid_search.best_score_,
        )
        if make_plots:
            plots.plot_confusion_matrix(self.grid_search.best_estimator_, self.data, self.target)
            plots.plot_params_scores(cv_results)

    def get_results(self):
        return self.result


def _make_params_key(params):
    return f"Metric: {params['model__metric']}, n neighbors: {params['model__n_neighbors']}"
