import sys
import os
import glob
import time
import random
import argparse

import numpy as np
import torch
import torchvision

from dataset import CINIC10
from models import *


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def get_dataset(args):
    if args.dataset == 'cifar10':
        args.data_mean = (0.4914, 0.4822, 0.4465)
        args.data_std = (0.2023, 0.1994, 0.2010)
        args.num_classes = 10
        
        return torchvision.datasets.CIFAR10
    elif args.dataset == 'cifar100':
        args.data_mean = (0.5071, 0.4867, 0.4408)
        args.data_std = (0.2675, 0.2565, 0.2761)
        args.num_classes = 100

        return torchvision.datasets.CIFAR100
    elif args.dataset == 'cinic-10':
        args.data_mean = (0.47889522, 0.47227842, 0.43047404)
        args.data_std = (0.24205776, 0.23828046, 0.25874835)
        args.num_classes = 10

        return CINIC10
    elif args.dataset == 'mnist':
        args.data_mean = (0.1307,)
        args.data_std = (0.3081,)
        args.num_classes = 10

        return torchvision.datasets.MNIST
    else:
        raise NotImplementedError()


def load_model(args):
    if args.net == 'wrn28-10':
        net = Wide_ResNet(28, 10, 0.3, args.num_classes)
    elif args.net == 'res18':
        net = ResNet18(num_classes=args.num_classes)
    elif args.net == 'res34':
        net = ResNet34(num_classes=args.num_classes)
    elif args.net == 'res50':
        net = ResNet50(num_classes=args.num_classes)
    elif args.net == 'vgg13':
        net = VGG('VGG13', num_classes=args.num_classes)
    elif args.net == 'vgg16':
        net = VGG('VGG16', num_classes=args.num_classes)
    elif args.net == 'vgg19':
        net = VGG('VGG19', num_classes=args.num_classes)
    elif args.net == 'mobilenetv2':
        net = MobileNetV2(num_classes=args.num_classes)
    else:
        raise NotImplementedError()

    return net


def load_shadow_models(args):
    shadow_net = args.shadow_net

    ckpt_paths = glob.glob(f'./checkpoints/{args.dataset}/shadow/{shadow_net}/*.pth')
    ckpt_paths.sort()

    if len(ckpt_paths) < args.num_shadow:
        raise ValueError(f'Expected {args.num_shadow} shadow models, got: {len(ckpt_paths)}')
    arch = os.path.splitext(os.path.basename(ckpt_paths[0]))[0].split('_')[0]
    
    models = []
    for p in ckpt_paths[:args.num_shadow]:
        checkpoint = torch.load(p, map_location='cpu')
        net = load_model(argparse.Namespace(net=arch, num_classes=args.num_classes))
        net.load_state_dict(checkpoint['model'])
        net.eval()
        net.to(args.device)
        models.append(net)
    return models

try:
	_, term_width = os.popen('stty size', 'r').read().split()
except:
	term_width = 80
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f