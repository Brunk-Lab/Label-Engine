import torch
import math
from engine.core.utils.log import logger


def get_optimizer(model, optimizer_params):
    params = []
    opt_name = optimizer_params['name']
    base_lr = optimizer_params['lr']
    for name, param in model.named_parameters():
        param_group = {'params': [param]}
        if not param.requires_grad:
            params.append(param_group)
            continue

        if not math.isclose(getattr(param, 'lr_mult', 1.0), 1.0):
            logger.info(f'Applied lr_mult={param.lr_mult} to "{name}" parameter.')
            param_group['lr'] = param_group.get('lr', base_lr) * param.lr_mult

        params.append(param_group)

    betas = optimizer_params['betas']
    eps = optimizer_params['eps']
    optimizer = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW
    }[opt_name.lower()](params, lr=base_lr, betas=betas, eps=eps)

    return optimizer
