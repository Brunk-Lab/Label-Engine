from core.utils.exp_imports.default import *

MODEL_NAME = 'unet_2048x2448_ecDNA'


def main(cfg: Dict) -> None:
    model = build_model(img_size=(2048, 2448))
    train(model, cfg)


def build_model(img_size: Tuple[int, int]) -> UNet:
    num_in_channels = 1
    num_out_channels = 2

    model = UNet(num_in_channels, num_out_channels)
    max_stride = model.max_stride
    assert img_size[0] % max_stride == 0 and img_size[1] % max_stride == 0
    return model

def train(model: UNet, cfg: Dict) -> None:
    
    model.to(cfg.device)

    loss_cfg = edict()

    trainer = AutoTrainer()
    trainer.run(num_epochs=500, validation=False)