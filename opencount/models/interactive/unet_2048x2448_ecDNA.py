from core.utils.exp_imports.default import *

MODEL_NAME = 'unet_2048x2448_ecDNA'


def main(cfg):
    model = build_model(img_size=(2048, 2448))
    train(model, cfg)


def build_model(img_size):
    pass

def train(model, cfg) -> None:
    pass