import copy

import random

import numpy as np
import torch
import torch.nn as nn


def get_log_logits_torch(logits, y):
    logits = logits - torch.max(logits, dim=-1, keepdims=True)[0]
    logits = torch.exp(logits)
    logits = logits / torch.sum(logits, dim=-1, keepdims=True)

    y_true = logits[np.arange(logits.size(0)), y]
    logits_masked = logits.clone()
    logits_masked[np.arange(logits.size(0)), y] = 0
    y_wrong = torch.sum(logits_masked, dim=-1)
    logits = (torch.log(y_true+1e-45) - torch.log(y_wrong+1e-45))

    return logits


def get_attack_loss(loss):
    if loss == 'target_logits' or loss == 'target_logits_log':
        return nn.MSELoss()
    else:
        raise NotImplementedError()


def initialize_poison(inputs):
    """Official version proposes several initializations.
      We choose what works best: from target image.
    """
    init = copy.deepcopy(inputs)
    init.requires_grad = True

    return init


def calculate_loss(x, y, shadow_models, criterion, args):
    """Official version proposes several ways to calculate loss.
      We choose what works best: `target_logits` or `target_logits_log` loss versions.

    Note: Here, we could've simplified some of the computations, but don't do so to keep
      with official impl. and ease an extension to other losses.
    """
    if 'target_logits' in args.canary_loss:
        with torch.no_grad():
            tmp_outputs = shadow_models[0](x)
        y_out = torch.zeros(tmp_outputs.shape, device=args.device, dtype=tmp_outputs.dtype)
        y_out[np.arange(y_out.size(0)), y] += args.canary_target_logit
        y_out = y_out[np.arange(y_out.size(0)), y]
    
    out_loss = 0.
    for curr_model in shadow_models:
        outputs = curr_model(x)
        
        if args.canary_loss == 'target_logits':
            outputs = outputs[np.arange(outputs.size(0)), y]
        elif args.canary_loss == 'target_logits_log':
            outputs = get_log_logits_torch(outputs, y)
        
        curr_loss = criterion(outputs, y_out)
        out_loss += curr_loss
    
    out_loss = out_loss / len(shadow_models)

    return out_loss


def generate_canaries(inputs, targets, shadow_models, args):
    """https://arxiv.org/pdf/2210.10750.pdf
    """
    criterion = get_attack_loss(args.canary_loss)

    x = initialize_poison(inputs.to(args.device))
    y = targets.to(args.device)

    dm = torch.tensor(args.data_mean)[None, :, None, None].to(args.device)
    ds = torch.tensor(args.data_std)[None, :, None, None].to(args.device)

    optimizer = torch.optim.AdamW([x], lr=0.05, weight_decay=0.001)

    for step in range(args.canary_iter):
        curr_shadow_models = random.sample(shadow_models, args.canary_stochastic_k)

        loss = calculate_loss(x, y, curr_shadow_models, criterion, args)
        optimizer.zero_grad()

        if loss != 0:
            x.grad, = torch.autograd.grad(loss, [x])
        optimizer.step()

        # projection
        with torch.no_grad():
            if args.canary_epsilon:
                x_diff = x.data - inputs.data
                x_diff.data = torch.max(torch.min(x_diff, args.canary_epsilon /
                                                        ds / 255), -args.canary_epsilon / ds / 255)
                x_diff.data = torch.max(torch.min(x_diff, (1 - dm) / ds - inputs), -dm / ds - inputs)
                x.data = inputs.data + x_diff.data
            else:
                x.data = torch.max(torch.min(x, (1 - dm) / ds), -dm / ds)
        
        if loss <= 23.0:  # default value used in official repo.
            break

    return x.detach()
