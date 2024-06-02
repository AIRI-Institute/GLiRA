import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Subset
import torchvision.transforms as transforms

import argparse

from utils import set_random_seed, get_dataset, load_model, progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--opt', default='sgd')
parser.add_argument('--net', default='res18')
parser.add_argument('--target_net', default=None, type=str)
parser.add_argument('--num_shadow', default=128, type=int)
parser.add_argument('--shadow_idx', default=None, type=int)
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

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0

size = imsize

tv_dataset = get_dataset(args)

print('==> Preparing data..')
if args.dataset == 'mnist':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(args.data_mean, args.data_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(args.data_mean, args.data_std),
    ])
else:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(args.data_mean, args.data_std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(args.data_mean, args.data_std),
    ])

# Prepare dataset
trainset = tv_dataset(root=f'./data', train=True, download=True, transform=transform_train)
dataset_size = len(trainset)

# set random seed
set_random_seed(args.seed)

# get shadow dataset
if not os.path.exists(f'./data_split/{args.dataset}'):
    print(f'No shadow/target indices created for {args.dataset} yet, assigning...')
    os.makedirs(f'./data_split/{args.dataset}')

    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    target_indices, shadow_indices = indices[:args.per_model_dataset_size], indices[args.per_model_dataset_size:]
    target_indices.sort()

    list_shadow_indices = []
    for _ in range(args.num_shadow):
        indices = np.random.choice(shadow_indices, size=args.per_model_dataset_size, replace=False)
        indices.sort()
        list_shadow_indices.append(indices)
    list_shadow_indices = np.stack(list_shadow_indices, axis=0)
    
    np.save(f'./data_split/{args.dataset}/target_indices.npy', target_indices)
    np.save(f'./data_split/{args.dataset}/shadow_indices.npy', list_shadow_indices)

if args.shadow_idx is None:
    print('Training will be performed for target model.')
    indices = np.load(f'./data_split/{args.dataset}/target_indices.npy')
else:
    print(f'Training will be performed for shadow model: {args.shadow_idx}.')
    indices = np.load(f'./data_split/{args.dataset}/shadow_indices.npy')[args.shadow_idx]
    
if len(indices) < args.per_model_dataset_size:
    target_indices = np.load(f'./data_split/{args.dataset}/target_indices.npy')
    all_indices = list(range(dataset_size))
    free_indices = list(set(all_indices) - set(list(target_indices)) - set(list(indices)))
    np.random.shuffle(free_indices)
    add_indices = free_indices[:(args.per_model_dataset_size - len(indices))]
    indices = np.concatenate([indices, np.array(add_indices)])
    print(f'Extended dataset size to {args.per_model_dataset_size}; added {len(add_indices)}.')

indices = indices[:args.per_model_dataset_size]

trainset = Subset(trainset, indices)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=16)

testset = tv_dataset(root=f'./data', train=False, download=True, transform=transform_test)
testsize = min(10000, len(testset))
test_indices = np.random.choice(len(testset), testsize, replace=False)
testset = Subset(testset, test_indices)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=16)

print(f'Train dataset size = {len(trainset)}; Test dataset size = {len(testset)}')
time.sleep(3)

# Model factory..
print('==> Building model..')
net = load_model(args)

if args.target_net is not None:
    if args.lambd == 1.0 and not args.mse_distillation:
        raise ValueError('Warning: target network loaded but the kldiv loss coefficient is 0.')

    print(f'==> Building target model for distillation: {args.target_net}.')
    target_args = argparse.Namespace(net=args.target_net.split('_')[0], num_classes=args.num_classes)
    checkpoint = torch.load(os.path.join(f'./checkpoints/{args.dataset}/target/{args.target_net}.pth'))
    target_net = load_model(target_args)
    target_net.load_state_dict(checkpoint['model'])
    target_net.cuda()
    target_net.eval()
elif args.lambd != 1:
    raise ValueError('Warning: kldiv loss coefficient > 0, but no target model loaded.')
else:
    pass

# Losses: CrossEntropy and KLDiv for distillation
ce_criterion = nn.CrossEntropyLoss()
kldiv_criterion = nn.KLDivLoss(reduction='batchmean')
mse_criterion = nn.MSELoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
# use cosine scheduling
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

if args.mse_distillation:
    print('Using MSE for shadow model training!')

def train(epoch):
    if epoch < args.warmup_epochs:
        print('\nWarmup Epoch: %d' % epoch)
    else:
        print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)

            if epoch < args.warmup_epochs:
                loss = ce_criterion(outputs, targets)
            else:
                if args.mse_distillation:
                    with torch.no_grad():
                        teacher_outputs = target_net(inputs)
                        
                    if args.mse_blackbox:
                        teacher_probs = F.softmax(teacher_outputs, dim=1)
                        
                        # obtain teacher logits on the assumption that sum(teacher_outputs) = 0
                        teacher_probs = torch.clamp(teacher_probs, 1e-32, None)
                        log_probs = torch.log(teacher_probs)
                        
                        C = -torch.mean(log_probs, 1).unsqueeze(1)
                        teacher_logits = log_probs + C
                        
                        loss = mse_criterion(outputs, teacher_logits)
                    else:
                        loss = mse_criterion(outputs, teacher_outputs)
                else:
                    if args.lambd < 1.0:
                        with torch.no_grad():
                            teacher_outputs = target_net(inputs)

                        kldiv_loss = kldiv_criterion(
                            F.log_softmax(outputs / args.tau, dim=1),
                            F.softmax(teacher_outputs / args.tau, dim=1)
                        )
                    else:
                        kldiv_loss = 0.
                    
                    ce_loss = ce_criterion(outputs, targets)
                    loss = args.lambd * ce_loss + (1. - args.lambd) * kldiv_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))      

    return train_loss/(batch_idx+1), 100.*correct/total


##### Validation
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = ce_criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return test_loss/(batch_idx+1), 100.*correct/total


net.cuda()

list_train_loss, list_train_acc = [], []
list_loss, list_acc = [], []
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    train_loss, train_acc = train(epoch)
    val_loss, val_acc = test(epoch)
    
    scheduler.step() # step cosine scheduling
    
    list_train_loss.append(train_loss)
    list_train_acc.append(train_acc)
    list_loss.append(val_loss)
    list_acc.append(val_acc)
list_loss, list_acc = np.array(list_loss), np.array(list_acc)

state = {
    "model": net.state_dict(),
    "train_loss": list_train_loss,
    "train_acc": list_train_acc,
    "val_loss": list_loss,
    "val_acc": list_acc
}

if args.shadow_idx is None:
    os.makedirs(f'./checkpoints/{args.dataset}/target', exist_ok=True)
    if args.show_samples:
        torch.save(state, f'./checkpoints/{args.dataset}/target/{args.net}_samples{args.per_model_dataset_size}.pth')
    else:
        torch.save(state, f'./checkpoints/{args.dataset}/target/{args.net}.pth')
else:
    d = f'./checkpoints/{args.dataset}/shadow/{args.net}'
    if args.target_net is not None:
        if args.mse_distillation:
            if args.mse_blackbox:
                d = d + '_bbMSEdis_' + args.target_net
            else:
                d = d + f'_MSEdis_' + args.target_net
        elif args.tau != 1:
            d = d + f'_{(1.0 - args.lambd):.1f}dis_{args.tau:.1f}tau_' + args.target_net
        else:
            d = d + f'_{(1.0 - args.lambd):.1f}dis_' + args.target_net
    if args.show_samples:
        d = d + f'_samples{args.per_model_dataset_size}'
            
    os.makedirs(d, exist_ok=True)
    torch.save(state, os.path.join(d, args.net + '_' + str(args.shadow_idx) + '.pth'))
