import torch
from typing import Tuple, Dict
from easydict import EasyDict as edict

from opencount.core.data.ecDNA import ecDNADataset
from opencount.core.engine.auto_trainer import AutoTrainer
from opencount.core.engine.icount_trainer import ICTrainer
from opencount.core.model.auto_unet_model import UNet
from opencount.core.model.icount_unet_model import iUNet
from opencount.core.utils.normalizer import FixedNormalizer, AdaptiveNormalizer