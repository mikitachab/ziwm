from collections import namedtuple

ParamsScore = namedtuple('ParamsScore', 'metric, n_neighbors, k, scores')


def get_params_scores(cv_results, n_attrs):
    return [
        ParamsScore(
            metric=params_chunk.iloc[0]['param_model__metric'],
            n_neighbors=params_chunk.iloc[0]['param_model__n_neighbors'],
            k=params_chunk['param_selector__k'].to_list(),
            scores=params_chunk['mean_test_score'].to_list()
        )
        for params_chunk in df_chunks(cv_results, n_attrs)
    ]


def df_chunks(df, chunksize):
    return (df[i:i + chunksize] for i in range(0, df.shape[0], chunksize))
