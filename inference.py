import os
import pickle
import argparse
import re

import pandas as pd
import torch

from inference_utils import evaluate, cache_evaluate


def legend_dict(x):
    if re.match('^[0-9a-zA-Z\-]+_\d.\ddis_\d+.\dtau_[0-9a-zA-Z\-]+$', x):
        coef, tau = re.findall('^[0-9a-zA-Z\-]+_(\d.\d)dis_(\d+.\d)tau_[0-9a-zA-Z\-]+$', x)[0]
        return f'{coef} {tau} GLiRA (KL)'
    if re.match('^[0-9a-zA-Z\-]+_\d.\ddis_[0-9a-zA-Z\-]+$', x):
        coef = re.findall('^[0-9a-zA-Z\-]+_(\d.\d)dis_[0-9a-zA-Z\-]+$', x)[0]
        return f'{coef} GLiRA (KL)'
    if re.match('^[0-9a-zA-Z\-]+$', x):
        return 'LiRA'
    if re.match('^[0-9a-zA-Z\-]+_MSEdis_[0-9a-zA-Z\-]+$', x):
        return 'GLiRA (MSE)'
    else:
        return x


def main(args):
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    args.device = device

    if args.cache_data:
        values, metrics = cache_evaluate(args)
    else:
        values, metrics = evaluate(args)

    if args.gen_canary:
        method = 'Canary'
    elif args.evaluation_type == 'calibration':
        method = 'Calibration'
    else:
        method = legend_dict(args.shadow_net)

    experiment = {
        'method': [method],
        'shadow_net': [args.shadow_net.split('_')[0]],
        'target_net': [args.target_net],
        'num_shadow': [args.num_shadow],
        'dataset': [args.dataset],
        'n_samples': [int(args.n_samples)],
        'objective': [args.evaluation_objective],
        'n_aug': [args.num_aug],
    }
    experiment.update({k: [v] for k, v in metrics.items()})
    experiment = pd.DataFrame(experiment)

    if os.path.exists(f'./results/{args.experiments_file}.pickle'):
        all_experiments = pd.read_pickle(f'./results/{args.experiments_file}.pickle')
        all_experiments = pd.concat([all_experiments, experiment], ignore_index=True)
    else:
        all_experiments = experiment

    all_experiments = all_experiments.drop_duplicates(
        subset=['method', 'dataset', 'target_net', 'shadow_net', 'num_shadow', 'n_samples', 'objective', 'n_aug'],
        keep='last')
    all_experiments = all_experiments.sort_values(['dataset', 'shadow_net', 'target_net', 'method'], 
                                                  ascending=True)
    all_experiments = all_experiments.reset_index(drop=True)
    all_experiments.to_pickle(f'./results/{args.experiments_file}.pickle')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch Inference')
    parser.add_argument('--shadow_net', type=str, required=True)
    parser.add_argument('--target_net', default=None, type=str)
    parser.add_argument('--num_shadow', default=128, type=int)
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--size', default=32, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_aug', default=1, type=int)
    parser.add_argument('--n_samples', default=20000, type=int)
    parser.add_argument('--evaluation_objective', default='stable_logit', type=str)
    parser.add_argument('--evaluation_type', default='lira', type=str)
    parser.add_argument('--fix_variance', action='store_true')
    parser.add_argument('--device', default=0, type=int)

    parser.add_argument('--gen_canary', action='store_true')
    parser.add_argument('--canary_loss', default='target_logits', type=str)
    parser.add_argument('--canary_target_logit', default=0, type=float)
    parser.add_argument('--canary_iter', default=30, type=int)
    parser.add_argument('--canary_stochastic_k', default=2, type=int)
    parser.add_argument('--canary_epsilon', default=1, type=float)

    parser.add_argument('--experiments_file', default='results', type=str)

    parser.add_argument('--cache_data', action='store_true')

    args = parser.parse_args()

    main(args)
