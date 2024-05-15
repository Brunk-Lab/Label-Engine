import torch


def get_scheduler(optimizer, scheduler_params):
    if scheduler_params['name'] == 'MultiStepLR':
        milestones = scheduler_params['milestones']
        gamma = scheduler_params['gamma']
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones,
            gamma
        )
        return scheduler
    else:
        raise ValueError('Undefined scheduler.')