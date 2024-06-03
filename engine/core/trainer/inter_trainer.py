import cv2
from datetime import datetime
import json
import math
import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from engine.core.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from engine.core.utils.vis import draw_probmap, draw_points
from engine.core.model.inter_base_model import interModel
from engine.core.utils.optimizer import get_optimizer
from engine.core.utils.scheduler import get_scheduler
from engine.core.utils.distributed import get_sampler
from engine.core.utils.simpleitk import convert_tensor_to_image, \
    get_num_connected_component
from engine.core.loss.focal_loss import FocalLoss


class InterTrainer(object):
    def __init__(
            self,
            model: interModel,
            cfg: Dict,
            trainset,
            valset,
            loss_func_params: Dict,
            optimizer_params: Dict,
            scheduler_params: Dict,
            image_dump_interval: int,
            checkpoint_interval: int,
            validation_interval: int,
    ) -> None:
        self.cfg = cfg
        self.is_master = self.cfg.local_rank == 0
        if self.is_master:
            logger.info(f'Training cases: {len(trainset)}')
            logger.info(f'Validation cases: {len(valset)}')

        self.train_data = DataLoader(
            trainset, batch_size=cfg.batch_size,
            sampler=get_sampler(trainset, shuffle=True, distributed=True),
            num_workers=cfg.workers,
            drop_last=True, pin_memory=True
        )

        self.val_data = DataLoader(
            valset, batch_size=1,
            sampler=get_sampler(valset, shuffle=False, distributed=True),
            num_workers=1,
            drop_last=False, pin_memory=True,
        )

        self.device = cfg.device
        self.model = model.to(self.device)
        self.model = load_weights(self.model, self.cfg.weights)

        self.model = DDP(self.model, device_ids=[cfg.gpu_ids[cfg.local_rank]],
                        broadcast_buffers=False)

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

        self.image_dump_interval = image_dump_interval
        self.checkpoint_interval = checkpoint_interval
        self.validation_interval = validation_interval

    def run(self, num_epochs, start_epoch=None, validation=False):
        if start_epoch is None:
            start_epoch = self.cfg.start_epoch

        if self.is_master:
            logger.info(f'Starting Epoch: {start_epoch}')
            logger.info(f'Total Epochs: {num_epochs}')

        for epoch in range(start_epoch, num_epochs):
            self.training(epoch)
            if validation and epoch % self.validation_interval == 0:
                self.validation(epoch)

    def training(self, epoch):
        self.model.train()

        self.train_data.sampler.set_epoch(epoch)
        for i, batch_data in enumerate(self.train_data):
            global_step = epoch * len(self.train_data) + i

            loss, batch_data, output = self.batch_forward(batch_data)

            self.optim.zero_grad()
            loss.backward()                
            self.optim.step()

            # gather losses from all devices
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)

            if self.is_master:
                loss /= dist.get_world_size()
                logger.info(f'Epoch {epoch}, batch {i+1}/{len(self.train_data)}, ' + \
                            f'step {global_step+1}, train_loss {loss.item():.5f}')

                if self.image_dump_interval > 0 and \
                    global_step % self.image_dump_interval == 0:
                    self.save_visualization(batch_data, output, global_step, 'train')

        if self.is_master:
            save_checkpoint(self.model, self.cfg.CHECKPOINTS_PATH, epoch=-1)

            if isinstance(self.checkpoint_interval, (list, tuple)):
                freq = [x for x in self.checkpoint_interval if x[0] <= epoch][-1][1]
            else:
                freq = self.checkpoint_interval

            if epoch % freq == 0:
                save_checkpoint(self.model, self.cfg.CHECKPOINTS_PATH, epoch=epoch)

        self.sched.step()

    def validation(self, epoch):
        val_metrics = {'thresholds': [0.3, 0.35, 0.4, 0.45, 0.5], 'data': {}}
        self.model.eval()
        for i, batch_data in enumerate(self.val_data):
            loss, batch_data, output = self.batch_forward(batch_data, validation=True)

            # gather data from all devices
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)

            batch_masks = [None for _ in range(self.cfg.world_size)]
            batch_preds = [None for _ in range(self.cfg.world_size)]
            batch_coords = [None for _ in range(self.cfg.world_size)]
            batch_names = [None for _ in range(self.cfg.world_size)]
            dist.all_gather_object(batch_masks, batch_data['instances'])
            dist.all_gather_object(batch_preds, output['instances'])
            dist.all_gather_object(batch_coords, batch_data['coords'])
            dist.all_gather_object(batch_names, batch_data['image_names'])

            if self.is_master:
                loss /= dist.get_world_size()
                logger.info(f'Epoch {epoch}, batch {i+1}/{len(self.val_data)}, ' + \
                            f'val_loss {loss.item():.4f}')

                # save validation results
                metrics = self.batch_metrics(batch_masks, batch_preds, batch_coords, 
                                             batch_names, val_metrics['thresholds'])
                val_metrics['data'].update(metrics)

        if self.is_master:
            thresholds = val_metrics['thresholds']
            normal_cases_nr10 = [0 for _ in range(len(thresholds))]
            normal_cases_nr20 = [0 for _ in range(len(thresholds))]
            mae = [0 for _ in range(len(thresholds))]
            mse = [0 for _ in range(len(thresholds))]
            for image_name in val_metrics['data']:
                for i, threshold in enumerate(thresholds):
                    metric, n_pred, n_gt = val_metrics['data'][image_name][threshold]
                    mae[i] += abs(n_pred - n_gt)
                    mse[i] += (n_pred - n_gt)**2
                    if abs(metric - 1.0) <= 0.1:
                        normal_cases_nr10[i] += 1
                    if abs(metric - 1.0) <= 0.2:
                        normal_cases_nr20[i] += 1

            report = val_metrics['report'] = {}
            report['total_cases'] = num_cases = len(val_metrics['data'])
            report['metrics'] = {
                'NR_10': [val / num_cases for val in normal_cases_nr10],
                'NR_20': [val / num_cases for val in normal_cases_nr20],
                'MAE': [val / num_cases for val in mae],
                'MSE': [math.sqrt(val / num_cases) for val in mse],
            } if num_cases > 0 else {}

            now = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            with open(f'{self.cfg.VIS_PATH}/val_epoch_{epoch}_{now}.json', 'w') as fp:
                json.dump(val_metrics, fp)

    def batch_forward(self, batch_data, validation=False):

        with torch.set_grad_enabled(not validation):
            batch_data = {k: v if isinstance(v, list) else v.to(self.device) \
                          for k, v in batch_data.items()}
            image, mask = batch_data['images'], batch_data['instances']
            points = batch_data['points']

            image_feats = self.model.module.get_image_feats(image)
            prev_mask = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]

            prompts = {'points': points, 'prev_mask': prev_mask}
            prompt_feats = self.model.module.get_prompt_feats(prompts)
            output = self.model(image_feats, prompt_feats)

            # proceed with more interactions
            # TO BE Done

            pred = output['instances'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(-1, pred.shape[-1])

            mask = mask.permute(0, 2, 3, 1).contiguous()
            mask = mask.view(-1, mask.shape[-1])

            selected_sample_indices = torch.nonzero(mask[:, 0] <= 1).squeeze()
            mask = torch.index_select(mask, 0, selected_sample_indices)
            pred = torch.index_select(pred, 0, selected_sample_indices)

            train_loss = self.loss_func(pred, mask)

        return train_loss, batch_data, output

    def batch_metrics(
        self, 
        batch_masks, 
        batch_preds, 
        batch_coords, 
        batch_names, 
        thresholds
    ) -> Dict:
        metrics = {}
        for masks, preds, coords, names in zip(batch_masks, batch_preds, 
                                               batch_coords, batch_names):
            for i, name in enumerate(names):
                mask = masks[i][0] # To be done
                pred = preds[i][1]
                pred = convert_tensor_to_image(pred, dtype=np.float32)
                num_gt = int(coords[i])

                metrics[name] = {}
                for thr in thresholds:
                    num_pred = int(get_num_connected_component(pred > thr, 1))
                    metric = num_pred / num_gt
                    metrics[name][thr] = (metric, num_pred, num_gt)

        return metrics

    def save_visualization(
        self, 
        batch_data, 
        output: Dict, 
        global_step, 
        prefix
    ) -> None:
        output_images_path = self.cfg.VIS_PATH / prefix
        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f'{global_step:06d}'

        image_names = batch_data['image_names']
        images = batch_data['images']
        points = batch_data['points']
        gt_masks = batch_data['instances']
        pred_masks = output['instances']

        gt_masks = gt_masks.cpu().numpy()
        gt_mask = np.squeeze(gt_masks[0], axis=0)

        pred_masks = pred_masks.detach().cpu().numpy()
        pred_mask = pred_masks[0, 1]

        points = points.detach().cpu().numpy()
        points = points[0]

        image = images.cpu().numpy() * 255
        image = image[0].transpose(1, 2, 0)

        image_w_pts = draw_points(image, points[:len(points) // 2], (0, 255, 0))
        image_w_pts = draw_points(image_w_pts, points[len(points) // 2:], (255, 0, 0))

        gt_mask = draw_probmap(gt_mask, norm=True)
        pred_mask = draw_probmap(pred_mask, norm=True)
        viz_image = np.hstack((image_w_pts, gt_mask, pred_mask)).astype(np.uint8)

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{image_name_prefix}_{suffix}.png'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        _save_image(f'seg_{image_names[0]}', viz_image[:, :, ::-1])


def load_weights(model: interModel, weights_path: str) -> interModel:
    if weights_path is not None:
        if os.path.isfile(weights_path):
            current_state_dict = model.state_dict()
            new_state_dict = torch.load(weights_path, map_location='cpu')['state_dict']
            current_state_dict.update(new_state_dict)
            model.load_state_dict(current_state_dict)
        else:
            raise RuntimeError(f"=> no checkpoint found at '{weights_path}'")

    return model


def save_checkpoint(model, chk_folder, epoch, verbose=False):
    chk_name = 'last_checkpoint.pth' if epoch < 0 else f'{epoch:03d}.pth'
    if not chk_folder.exists():
        chk_folder.mkdir(parents=True)
    
    chk_path = chk_folder / chk_name
    if verbose:
        logger.info(f'Save checkpoint to {str(chk_path)}')
    
    net = model.module
    torch.save({'state_dict': net.state_dict(), 'config': net._config}, str(chk_path))