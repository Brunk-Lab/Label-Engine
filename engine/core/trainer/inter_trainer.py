import cv2
from datetime import datetime
import json
import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Dict
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

        if cfg.multi_gpu:
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

            loss, splitted_batch_data, output = \
                self.batch_forward(batch_data, validation=False)

            self.optim.zero_grad()
            loss.backward()                
            self.optim.step()

            # gather losses from all devices
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)

            if self.is_master:
                loss /= dist.get_world_size()
                logger.info(f'Epoch {epoch}, batch {i}, train_loss {loss.item():.4f}')

                if self.image_dump_interval > 0 and \
                    global_step % self.image_dump_interval == 0:
                    self.save_visualization(
                        splitted_batch_data, output, global_step, prefix='train'
                    )

        if self.is_master:
            save_checkpoint(self.model, self.cfg.CHECKPOINTS_PATH, epoch=-1, 
                            multi_gpu=self.cfg.multi_gpu)

            if isinstance(self.checkpoint_interval, (list, tuple)):
                freq = [x for x in self.checkpoint_interval if x[0] <= epoch][-1][1]
            else:
                freq = self.checkpoint_interval

            if epoch % freq == 0:
                save_checkpoint(self.model, self.cfg.CHECKPOINTS_PATH, epoch=epoch,
                                multi_gpu=self.cfg.multi_gpu)

        self.sched.step()

    def validation(self, epoch):
        val_metrics = {}
        self.model.eval()
        for i, batch_data in enumerate(self.val_data):
            loss, outputs, info = self.batch_forward(batch_data, validation=True)

            # gather data from all devices
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)

            gathered_outputs = [None for _ in range(self.cfg.world_size)]
            gathered_info = [None for _ in range(self.cfg.world_size)]
            dist.all_gather_object(gathered_outputs, outputs)
            dist.all_gather_object(gathered_info, info)

            if self.is_master:
                loss /= dist.get_world_size()
                logger.info(f'Epoch {epoch}, batch {i}, val_loss {loss.item():.4f}')

                # save validation results
                metrics = self.get_validation_metrics(gathered_outputs, gathered_info)
                val_metrics.update(metrics)

        if self.is_master:
            num_normal_cases = 0
            metrics_sum = 0.
            for key in val_metrics.keys():
                if val_metrics[key][0] <= 1.:
                    num_normal_cases += 1
                    metrics_sum += val_metrics[key][0]

            now = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            with open(f'{self.cfg.VIS_PATH}/val_epoch_{epoch}_{now}.json', 'w') as fp:
                json.dump(val_metrics, fp)

            report_msg = f'Report: normal/total: {num_normal_cases}/{len(val_metrics)}, \
                metric: {metrics_sum / (max(1, num_normal_cases)):.4f}'
            logger.info(report_msg)


    def batch_forward(self, batch_data, validation=False):

        with torch.set_grad_enabled(not validation):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
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

            selected_sample_indices = torch.nonzero(mask[:, 0] >= 0).squeeze()
            mask = torch.index_select(mask, 0, selected_sample_indices)
            pred = torch.index_select(pred, 0, selected_sample_indices)

            train_loss = self.loss_func(pred, mask)

        return train_loss, batch_data, output

    def get_validation_metrics(self, gathered_outputs, gathered_info):
        metrics = {}
        for i in range(len(gathered_outputs)):
            image_name_list = gathered_info[i]['name']
            num_coords_list = gathered_info[i]['num_coords']

            for j in range(len(image_name_list)):
                image_prob = gathered_outputs[i][j][1].data
                image_prob = convert_tensor_to_image(image_prob, dtype=np.float)

                coords_pred = int(get_num_connected_component(image_prob > 0.4, 1))
                coords_gt = int(num_coords_list[j].numpy())
                metric = coords_pred / coords_gt
                metrics[image_name_list[j]] = [metric, coords_pred, coords_gt]

        return metrics

    def save_visualization(
        self, 
        splitted_batch_data, 
        output: Dict, 
        global_step, 
        prefix
    ) -> None:
        output_images_path = self.cfg.VIS_PATH / prefix
        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f'{global_step:06d}'

        images = splitted_batch_data['images']
        points = splitted_batch_data['points']
        gt_masks = splitted_batch_data['instances']
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

        gt_mask[gt_mask < 0] = 0.25
        gt_mask = draw_probmap(gt_mask)
        pred_mask = draw_probmap(pred_mask)
        viz_image = np.hstack((image_w_pts, gt_mask, pred_mask)).astype(np.uint8)

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{image_name_prefix}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        _save_image('instance_segmentation', viz_image[:, :, ::-1])


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


def save_checkpoint(model, chk_folder, epoch, verbose=False, multi_gpu=False):
    chk_name = 'last_checkpoint.pth' if epoch < 0 else f'{epoch:03d}.pth'
    if not chk_folder.exists():
        chk_folder.mkdir(parents=True)
    
    chk_path = chk_folder / chk_name
    if verbose:
        logger.info(f'Save checkpoint to {str(chk_path)}')
    
    net = model.module if multi_gpu else model
    torch.save({'state_dict': net.state_dict(), 'config': net._config}, str(chk_path))