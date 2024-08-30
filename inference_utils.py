import os
import glob
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, ConcatDataset
from torchvision import transforms

from tqdm import tqdm

from utils import set_random_seed, get_dataset, load_model, load_shadow_models
from dataset import NpyDataset
from canary_utils import generate_canaries
from score import get_logits, get_hinge, lira_offline, calibration


def load_trained_model(net: str, num_classes: int, ckpt_path: str):
    model_args = argparse.Namespace(net=net, num_classes=num_classes)
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    net = load_model(model_args)
    net.load_state_dict(ckpt['model'])
    net.eval()

    return net


def prepare_testset(args):
    tv_dataset = get_dataset(args)

    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(args.data_mean, args.data_std),
        ])
    else:
        if args.num_aug > 1:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(args.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(args.data_mean, args.data_std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(args.size),
                transforms.ToTensor(),
                transforms.Normalize(args.data_mean, args.data_std),
            ])
    
    
    # Take equal number of member/non-member samples for evaluation
    trainset = tv_dataset(root=f'./data', train=True, download=True, transform=transform)
    testset = tv_dataset(root=f'./data', train=False, download=True, transform=transform)
        
    # NOTE: We store results of each experiment in `./roc_curves/{DATASET NAME}`
    if not os.path.exists(f'./results/data_split/{args.dataset}'):
        os.makedirs(f'./results/data_split/{args.dataset}')

        target_indices = np.load(f'./data_split/{args.dataset}/target_indices.npy')

        member_indices = np.random.choice(target_indices, args.n_samples // 2, replace=False)
        nonmember_indices = np.random.choice(list(range(len(testset))), args.n_samples // 2, replace=False)

        np.save(f'./results/data_split/{args.dataset}/member_indices.npy', member_indices)
        np.save(f'./results/data_split/{args.dataset}/nonmember_indices.npy', nonmember_indices)
    else:
        member_indices = np.load(f'./results/data_split/{args.dataset}/member_indices.npy')[:args.n_samples // 2]
        nonmember_indices = np.load(f'./results/data_split/{args.dataset}/nonmember_indices.npy')[:args.n_samples // 2]
    
    testset = ConcatDataset([Subset(trainset, member_indices),
                             Subset(testset, nonmember_indices)])
    labels = np.array([1]*len(member_indices) + [0]*len(nonmember_indices))

    return testset, labels


def cache_eval_data(args, cache_dir) -> None:
    print('Caching dataset...')
    os.makedirs(cache_dir)
    
    testset, labels = prepare_testset(args)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)

    task_labels = np.concatenate([targets.tolist() 
                                 for (_, targets) in testloader])[:, None]

    for n_aug in range(args.num_aug):
        images = []
        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader), 
                                                total=len(testloader), 
                                                desc=f'Caching [{n_aug}/{args.num_aug - 1}]',
                                                leave=True):
            inputs = inputs.permute(0, 2, 3, 1).numpy()
            images.append(inputs)
        images = np.concatenate(images)
        np.save(os.path.join(cache_dir, f'data_batch_{n_aug}.npy'), images)
    np.save(os.path.join(cache_dir, 'task_labels.npy'), task_labels)
    np.save(os.path.join(cache_dir, 'labels.npy'), labels) 
    print(f'Successfully cached dataset at: {cache_dir}')


def get_evaluation_objectives(logits: np.ndarray, labels: np.ndarray, 
                              objective_type: str = 'stable_logit'):
    """
      Args:
        logits (np.ndarray[*, n_classes])
        labels (np.ndarray[*, 1])
      Returns:
        values (np.ndarray[*])
    
    """
    if objective_type == 'logit':
        values = get_logits(logits, labels, stable=False)
    elif objective_type == 'stable_logit':
        values = get_logits(logits, labels, stable=True)
    elif objective_type == 'ce':
        sz = logits.shape[:-1]
        values = nn.CrossEntropyLoss(reduction='none')(
            torch.from_numpy(logits.reshape(-1, logits.shape[-1])).to(torch.float32),
            torch.from_numpy(labels.reshape(-1)).long()
        ).numpy().reshape(*sz)
    elif objective_type == 'hinge':
        values = get_hinge(logits, labels)
    else:
        raise NotImplementedError(f'Unknown evaluation value: {objective_type}.')

    return values


def get_evaluation_metrics(
    args, 
    target_logits: np.ndarray, 
    shadow_logits: np.ndarray,
    labels: np.ndarray, 
    evaluation_type: str = 'lira'
):
    if evaluation_type == 'lira':
        fpr, tpr, auc, acc, low = lira_offline(target_logits, shadow_logits, labels,
                                               fix_variance=args.fix_variance)
    elif evaluation_type == 'calibration':
        fpr, tpr, auc, acc, low = calibration(target_logits, shadow_logits, labels)
    else:
        raise NotImplementedError(f'Unknown evaluation type: {evaluation_type}.')

    return fpr, tpr, auc, acc, low


def evaluate(args):
    set_random_seed(args.seed)

    device = args.device
    bs = int(args.bs)
    
    # 1. Prepare evaluation dataset
    testset, labels = prepare_testset(args)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=4)

    # 2. Extract evaluation values from target and shadow models
    target_model = load_trained_model(
        args.target_net.split('_')[0], args.num_classes,
        os.path.join(f'./checkpoints/{args.dataset}/target/{args.target_net}.pth')
    ).to(device)
    shadow_models = load_shadow_models(args)

    task_labels = np.concatenate([targets.tolist() 
                                  for (_, targets) in testloader])[:, None]

    target_objectives = []
    shadow_objectives = []
    for n_aug in range(args.num_aug):
        target_aug_logits = []
        shadow_aug_logits = [[] for _ in range(len(shadow_models))]
        
        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader), 
                                                 total=len(testloader), 
                                                 desc=f'Inference [{n_aug}/{args.num_aug - 1}]',
                                                 leave=True):
            inputs = inputs.to(device)

            if args.gen_canary:
                inputs = generate_canaries(inputs, targets, shadow_models, args)

            with torch.no_grad():
                logits = target_model(inputs).cpu().tolist()
            target_aug_logits.append(logits)

            for i, model in enumerate(shadow_models):
                with torch.no_grad():
                    logits = model(inputs).cpu().tolist()
                shadow_aug_logits[i].append(logits)
            
        target_aug_logits = np.concatenate(target_aug_logits)
        shadow_aug_logits = np.stack(
            [np.concatenate(logits) for logits in shadow_aug_logits]
        )

        target_aug_objectives = get_evaluation_objectives(
            target_aug_logits, 
            task_labels, 
            args.evaluation_objective
        )
        shadow_aug_objectives = get_evaluation_objectives(
            shadow_aug_logits,
            np.repeat(task_labels[None], len(shadow_models), 0),
            args.evaluation_objective
        )

        target_objectives.append(target_aug_objectives)
        shadow_objectives.append(shadow_aug_objectives)

    target_objectives = np.stack(target_objectives, axis=-1)
    shadow_objectives = np.swapaxes(np.stack(shadow_objectives, axis=-1), 0, 1)

    # 3. run specified score and metrics extraction.
    fpr, tpr, auc, acc, low = get_evaluation_metrics(args, target_objectives, shadow_objectives,
                                                     labels, args.evaluation_type)
    
    values = {
        'objective_type': args.evaluation_objective,
        'target_objectives': target_objectives,
        'shadow_objectives': shadow_objectives,
        'labels': labels
    }
    metrics = {
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc,
        'acc': acc,
        'tpr@fpr': low
    }

    return values, metrics


def cache_evaluate(args):
    set_random_seed(args.seed)

    device = args.device
    bs = int(args.bs)

    if args.dataset == 'cifar10':
        args.data_mean = (0.4914, 0.4822, 0.4465)
        args.data_std = (0.2023, 0.1994, 0.2010)
        args.num_classes = 10
    elif args.dataset == 'cinic-10':
        args.num_classes = 10
        args.data_mean = (0.47889522, 0.47227842, 0.43047404)
        args.data_std = (0.24205776, 0.23828046, 0.25874835)
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.data_mean = (0.5071, 0.4867, 0.4408)
        args.data_std = (0.2675, 0.2565, 0.2761)
    else:
        raise NotImplementedError()
    
    # 1. Extract evaluation values from target and shadow models.
    target_model = load_trained_model(
        args.target_net.split('_')[0], args.num_classes,
        os.path.join(f'./checkpoints/{args.dataset}/target/{args.target_net}.pth')
    ).to(device)
    shadow_models = load_shadow_models(args)

    # 2. Setup dataset path; get labels
    cache_dir = f'./eval_data/{args.dataset}'
    if os.path.exists(cache_dir):
        print(f'Using cached dataset at: {cache_dir}')
    else:
        cache_eval_data(args, cache_dir)

    task_labels = np.load(os.path.join(cache_dir, 'task_labels.npy'))
    labels = np.load(os.path.join(cache_dir, 'labels.npy'))

    target_objectives = []
    shadow_objectives = []
    for n_aug in range(args.num_aug):
        target_aug_logits = []
        shadow_aug_logits = [[] for _ in range(len(shadow_models))]

        testset = NpyDataset(cache_dir, n_aug)
        
        testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=4)
        
        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader), 
                                                 total=len(testloader), 
                                                 desc=f'Inference [{n_aug}/{args.num_aug - 1}]',
                                                 leave=True):
            inputs = inputs.to(device)
            
            if args.gen_canary:
                inputs = generate_canaries(inputs, targets, shadow_models, args)

            with torch.no_grad():
                logits = target_model(inputs).cpu().tolist()
            target_aug_logits.append(logits)

            for i, model in enumerate(shadow_models):
                with torch.no_grad():
                    logits = model(inputs).cpu().tolist()
                shadow_aug_logits[i].append(logits)
            
        target_aug_logits = np.concatenate(target_aug_logits)
        shadow_aug_logits = np.stack(
            [np.concatenate(logits) for logits in shadow_aug_logits]
        )

        target_aug_objectives = get_evaluation_objectives(
            target_aug_logits, 
            task_labels, 
            args.evaluation_objective
        )
        shadow_aug_objectives = get_evaluation_objectives(
            shadow_aug_logits,
            np.repeat(task_labels[None], len(shadow_models), 0),
            args.evaluation_objective
        )

        target_objectives.append(target_aug_objectives)
        shadow_objectives.append(shadow_aug_objectives)

    target_objectives = np.stack(target_objectives, axis=-1)
    shadow_objectives = np.swapaxes(np.stack(shadow_objectives, axis=-1), 0, 1)

    # 3. run specified score and metrics extraction.
    fpr, tpr, auc, acc, low = get_evaluation_metrics(args, target_objectives, shadow_objectives,
                                                     labels, args.evaluation_type)
    
    values = {
        'objective_type': args.evaluation_objective,
        'target_objectives': target_objectives,
        'shadow_objectives': shadow_objectives,
        'labels': labels
    }
    metrics = {
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc,
        'acc': acc,
        'tpr@fpr': low
    }

    return values, metrics
