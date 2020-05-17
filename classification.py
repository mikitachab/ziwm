from sklearn.neighbors import KNeighborsClassifier
import ilpd


from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


import pandas as pd


def make_experiment_result():
    data = ilpd.get_data(normalized=True)

    X = data.drop('Selector', axis=1).to_numpy()
    y = data['Selector'].to_numpy()

    n_features = X.shape[1]

    knn = KNeighborsClassifier()
    chi2_filter = SelectKBest(score_func=chi2)

    pipe = Pipeline(
        [
            ('chi2', chi2_filter),
            ('knn', knn)
        ]
    )

    param_grid = {
        'chi2__k': range(1, n_features + 1),
        'knn__n_neighbors': [1, 5, 10],
        'knn__metric': ['manhattan', 'euclidean']
    }

    cv = RepeatedKFold(n_splits=2, n_repeats=5)

    grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=cv, verbose=2)
    grid_search.fit(X, y)
    
    results = grid_search.cv_results_
    return pd.DataFrame(results)
