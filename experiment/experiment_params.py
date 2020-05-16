from math import floor

from .constants import CV_FOLDS, CV_REPEATS, METRICS, SMALLER_CLASS_INSTANCE_SIZE


def get_experiment_params(data):
    max_n_neighbors = SMALLER_CLASS_INSTANCE_SIZE * 2 + 1
    return ExperimentParams(data.columns.to_list(), max_n_neighbors, METRICS, CV_FOLDS)


class ExperimentParams:
    def __init__(self, attributes, max_n_neighbor, metrics, k_folds=2, class_number=2):
        self.attributes = attributes
        self.n_neighbors = self.__prepare_n_neighbors_list(floor(max_n_neighbor), class_number, k_folds)
        self.metrics = metrics
        self.k_folds = k_folds

    def get_param_grid(self):
        return {
            'model__n_neighbors': self.n_neighbors,
            'model__metric': self.metrics,
            'selector__k': range(1, len(self.attributes) + 1)
        }

    def get_cv_params(self):
        return self.k_folds, CV_REPEATS

    def __prepare_n_neighbors_list(self, max_n_neighbor, class_number, k_folds):
        n_neighbors_list = []

        if max_n_neighbor > k_folds:
            max_n_neighbor = floor(max_n_neighbor / k_folds)

        for i in range(1, max_n_neighbor):
            if i % class_number != 0:
                n_neighbors_list.append(i)
        return n_neighbors_list
