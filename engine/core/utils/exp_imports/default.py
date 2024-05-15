import torch
from typing import Tuple, Dict
from easydict import EasyDict as edict

from engine.core.data.ecDNA import ecDNADataset
from engine.core.trainer.auto_trainer import AutoTrainer
from engine.core.trainer.icount_trainer import ICTrainer
from engine.core.model.auto_unet_model import UNet
from engine.core.model.icount_unet_model import iUNet
from engine.core.utils.normalizer import FixedNormalizer, AdaptiveNormalizer