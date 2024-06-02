import os
import argparse

import subprocess


def check_existance(args, shadow_idx):
    d = f'./checkpoints/{args.dataset}/shadow/{args.net}'
    if args.target_net is not None:
        if args.full_black_box:
            d = d + f'_{(1.0 - args.lambd):.1f}dis_bb_' + args.target_net 
        elif args.mse_distillation:
            d = d + f'_MSEdis_' + args.target_net
            if args.lambd < 1.0:
                d = d + f'_{(1.0 - args.lambd):.1f}MSEdis_' + args.target_net
        elif args.tau != 1:
            d = d + f'_{(1.0 - args.lambd):.1f}dis_{args.tau:.1f}tau_' + args.target_net
        else:    
            d = d + f'_{(1.0 - args.lambd):.1f}dis_' + args.target_net
    if args.show_samples:
        d = d + f'_samples{args.per_model_dataset_size}'
        
    p = os.path.join(d, args.net + '_' + str(shadow_idx) + '.pth')

    if os.path.exists(p):
        return True
    
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training of Shadow models')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--opt', default='sgd')
    parser.add_argument('--net', default='res18')
    parser.add_argument('--target_net', default=None, type=str)
    parser.add_argument('--num_shadow', default=128, type=int)
    parser.add_argument('--start_shadow', default=0, type=int)
    parser.add_argument('--end_shadow', default=None, type=int)
    parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--per_model_dataset_size', default=20000, type=int)
    parser.add_argument('--bs', default=256, type=int)
    parser.add_argument('--size', default=32, type=int)
    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lambd', default=1.0, type=float)
    parser.add_argument('--tau', default=1.0, type=float)
    parser.add_argument('--mse_distillation', action='store_true')
    parser.add_argument('--mse_blackbox', action='store_true')
    parser.add_argument('--warmup_epochs', default=0, type=int)
    parser.add_argument('--show_samples', action='store_true')

    args = parser.parse_args()
    
    start_shadow = args.start_shadow
    del args.start_shadow

    end_shadow = args.end_shadow if args.end_shadow is not None \
        else args.num_shadow
    del args.end_shadow
    
    if start_shadow >= end_shadow:
        raise ValueError('start_shadow >= end_shadow.')
    
    call_arguments = ['python', 'train.py']
    for k, v in vars(args).items():
        if isinstance(v, bool):
            if v:
                call_arguments.append('--' + k)
        elif v is None:
            continue
        else:
            call_arguments.extend(['--' + k, str(v)])
    
    for i in range(start_shadow, end_shadow):
        if check_existance(args, i):
            print(f'Shadow model {i} already exists, skipping.', flush=True)
            continue

        call_arguments.extend(['--shadow_idx', str(i)])
        subprocess.call(call_arguments)
