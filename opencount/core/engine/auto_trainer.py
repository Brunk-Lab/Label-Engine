import os
import logging
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Dict

from opencount.core.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from opencount.core.model.auto_model import autoCountModel
from opencount.core.engine.optimizer import get_optimizer
from opencount.core.utils.distributed import get_dp_wrapper


class AutoTrainer(object):
    def __init__(
            self,
            model: autoCountModel,
            cfg: Dict,
            trainset,
            valset,
            optimizer: str='adam',
            optimizer_params=None,
            lr_scheduler=None,
    ) -> None:
        self.cfg = cfg
        self.is_master = self.cfg.local_rank == 0

        self.train_data = DataLoader(
            trainset,
            cfg.batch_size,
        )

        self.optim = get_optimizer(model, optimizer, optimizer_params)
        model = load_weights(model, self.cfg.weights)
        if cfg.multi_gpu:
            model = get_dp_wrapper()(model, device_ids=[cfg.gpu_ids[cfg.local_rank]],
                                     find_unused_parameters=True)

        self.device = cfg.device
        self.model = model.to(self.device)
        self.lr = optimizer_params['lr']

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)

    def training(self, epoch):
        pass

        if self.is_master:
            pass

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

    def validation(self, epoch):
        pass

    def batch_forward(self, batch_data, validation=False):
        pass

    def add_loss(self, loss_name):
        pass


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