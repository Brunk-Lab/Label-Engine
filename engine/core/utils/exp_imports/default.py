import torch
from typing import Tuple, Dict
from easydict import EasyDict as edict
from albumentations import *

from engine.core.data.datasets.ecDNA import ecDNADataset
from engine.core.data.datasets.roi import ROIDataset
from engine.core.data.transforms import ResizeLongestSide, UniformRandomResize, \
    remove_image_only_transforms
from engine.core.data.points_sampler import MultiPointSampler
from engine.core.trainer.inter_trainer import InterTrainer
from engine.core.trainer.seg_trainer import SegTrainer
from engine.core.model.auto_unet_model import autoUNet
from engine.core.model.inter_unet_model import interUNet
from engine.core.utils.normalizer import FixedNormalizer, AdaptiveNormalizer