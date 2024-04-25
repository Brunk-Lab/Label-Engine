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


    trainset = ecDNADataset(
        imlist_file='/playpen-raid2/qinliu/data/ecDNA/datasets/train_0422_2024.txt',
        labels={'foreground': 255},
        spacing=[1.0, 1.0],
        crop_size=(2048, 2448),
        sampling_method='CENTER',
        random_translation=[10, 10],
        random_scale=[0.8, 1.2],
        random_hori_flip=True,
        random_vert_flip=True,
        interpolation='LINEAR',
        crop_normalizers=[AdaptiveNormalizer(clip_sigma=5)],
        split='train'
    )

    loss_func_params = {'name': 'Focal', 'alpha': (0.5, 0.5), 'class_num': 2}
    optimizer_params = {'name': 'adam', 'lr': 5e-5, 'betas': (0.9, 0.999), 'eps': 1e-8}
    scheduler_params = {'name': 'MultiStepLR', 'milestones': [50, 90], 'gamma': 0.1}

    trainer = AutoTrainer(
        model,
        cfg,
        trainset,
        valset=None,
        loss_func_params=loss_func_params,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        image_dump_interval=1000,
        checkpoint_interval=200,
    )
    trainer.run(num_epochs=1001, validation=False)