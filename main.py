#!/usr/bin/env python3
import argparse

from features_ranking import make_features_ranking
from experiment import prepare_experiment as pe
import ilpd


def argparse_setup():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser('prepare-data').set_defaults(func=ilpd.prepare_data)

    exp_parser = subparsers.add_parser('run')
    exp_parser.set_defaults(func=run_experiment)
    exp_parser.add_argument('--make-plots', action='store_true')

    ranking_parser = subparsers.add_parser('features-ranking')
    ranking_parser.set_defaults(func=rank_features)
    ranking_parser.add_argument('--latex', action='store_true')
    return parser


def main():
    parser = argparse_setup()
    args = parser.parse_args()

    if 'func' in args:
        passed_args = omit(vars(args), 'func')
        if passed_args:
            args.func(**passed_args)
        else:
            args.func()
    else:
        parser.print_help()


def run_experiment(make_plots=False):
    exp = pe(ilpd.get_data(normalized=True))
    exp.run_experiment(make_plots=make_plots)
    result = exp.get_results()
    print('best score:', result.best_score)
    print('best params:', result.best_params)


def rank_features(latex=False):
    data = ilpd.get_data(normalized=True)
    x = data.drop('Selector', axis=1)
    y = data['Selector']
    features_ranking = make_features_ranking(x, y)

    if latex:
        print(features_ranking.to_latex())
    else:
        print(features_ranking)


def omit(dict_, *omit_keys):
    return {key: value for key, value in dict_.items() if key not in omit_keys}


if __name__ == '__main__':
    main()
