import os
import logging
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Dict
from tqdm import tqdm
from torch import distributed as dist

from opencount.core.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from opencount.core.model.auto_model import autoCountModel
from opencount.core.utils.optimizer import get_optimizer
from opencount.core.utils.scheduler import get_scheduler
from opencount.core.utils.distributed import get_dp_wrapper, get_sampler
from opencount.core.loss.focal_loss import FocalLoss


class AutoTrainer(object):
    def __init__(
            self,
            model: autoCountModel,
            cfg: Dict,
            trainset,
            valset,
            loss_func_params: Dict,
            optimizer_params: Dict,
            scheduler_params: Dict,
    ) -> None:
        self.cfg = cfg
        self.is_master = self.cfg.local_rank == 0

        self.train_data = DataLoader(
            trainset, cfg.batch_size,
            sampler=get_sampler(trainset, shuffle=True, distributed=True),
            num_workers=cfg.workers,
            drop_last=True, pin_memory=True
        )

        self.device = cfg.device
        self.model = model.to(self.device)

        if cfg.multi_gpu:
            self.model = get_dp_wrapper()(
                self.model, 
                device_ids=[cfg.gpu_ids[cfg.local_rank]],
                find_unused_parameters=False
            )

        self.model = load_weights(self.model, self.cfg.weights)
        self.optim = get_optimizer(self.model, optimizer_params)
        self.sched = get_scheduler(self.optim, scheduler_params)
        if cfg.start_epoch > 0:
            for _ in range(cfg.start_epoch):
                self.sched.step()

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        
        if loss_func_params['name'] == 'Focal':
            class_num = loss_func_params['class_num']
            alpha = loss_func_params['alpha']
            self.loss_func = FocalLoss(class_num, alpha, use_gpu=True)

    def run(self, num_epochs, start_epoch=None, validation=False):
        if start_epoch is None:
            start_epoch = self.cfg.start_epoch

        if self.is_master:
            logger.info(f'Starting Epoch: {start_epoch}')
            logger.info(f'Total Epochs: {num_epochs}')

        for epoch in range(start_epoch, num_epochs):
            self.training(epoch)
            if validation:
                self.validation(epoch)

    def training(self, epoch):
        self.model.train()

        self.train_data.sampler.set_epoch(epoch)
        tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=100) \
            if self.is_master else self.train_data

        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.train_data) + i

            loss = self.batch_forward(batch_data)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)

            if self.is_master:
                loss /= dist.get_world_size()
                lr = self.sched.get_lr()[-1]
                tbar.set_description(
                    f'Epoch {epoch}, training loss {loss.item():.4f}, lr {lr:.4f}'
                )

    def validation(self, epoch):
        pass

    def batch_forward(self, batch_data, validation=False):

        with torch.set_grad_enabled(not validation):
            crops, masks, frames, filenames = batch_data
            crops, masks = crops.to(self.device), masks.to(self.device)

            outputs = self.model(crops)
            outputs = outputs.permute(0, 2, 3, 1).contiguous()
            outputs = outputs.view(-1, outputs.shape[-1])

            masks = masks.permute(0, 2, 3, 1).contiguous()
            masks = masks.view(-1, masks.shape[-1])

            selected_sample_indices = torch.nonzero(masks[:, 0] >= 0).squeeze()
            masks = torch.index_select(masks, 0, selected_sample_indices)
            outputs = torch.index_select(outputs, 0, selected_sample_indices)

            train_loss = self.loss_func(outputs, masks)

        return train_loss


def load_weights(model: autoCountModel, weights_path: str) -> autoCountModel:
    if weights_path is not None:
        if os.path.isfile(weights_path):
            current_state_dict = model.state_dict()
            new_state_dict = torch.load(weights_path, map_location='cpu')['state_dict']
            current_state_dict.update(new_state_dict)
            model.load_state_dict(current_state_dict)
        else:
            raise RuntimeError(f"=> no checkpoint found at '{weights_path}'")

    return model