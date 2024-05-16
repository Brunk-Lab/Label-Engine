from core.utils.exp_imports.default import *

MODEL_NAME = 'unet_2048x2448_ecDNA'


def main(cfg: Dict) -> None:
    model = build_model(img_size=(2048, 2448))
    train(model, cfg)


def build_model(img_size: Tuple[int, int]) -> autoUNet:
    num_in_channels = 1
    num_out_channels = 2

    model = autoUNet(num_in_channels, num_out_channels)
    max_stride = model.max_stride
    assert img_size[0] % max_stride == 0 and img_size[1] % max_stride == 0
    return model


def train(model: autoUNet, cfg: Dict) -> None:

    augmentation_params = {
        'random_translation': [10, 10],
        'random_scale': [0.8, 1.2],
        'random_hori_flip': True,
        'random_vert_flip': True
    }

    trainset = ecDNADataset(
        dataset_path=cfg.ECDNA_PATH,
        split='train',
        crop_size=(2048, 2448),
        normalizer=AdaptiveNormalizer(clip_sigma=5),
        augmentation_params=augmentation_params,
    )

    valset = ecDNADataset(
        dataset_path=cfg.ECDNA_PATH,
        split='val',
        crop_size=(2048, 2448),
        normalizer=AdaptiveNormalizer(clip_sigma=5)
    )

    loss_func_params = {'name':'Focal', 'alpha':(0.5, 0.5), 'class_num':2}
    optimizer_params = {'name':'adam', 'lr':5e-5, 'betas':(0.9, 0.999), 'eps':1e-8}
    scheduler_params = {'name':'MultiStepLR', 'milestones':[500, 800], 'gamma':0.2}

    trainer = AutoTrainer(
        model,
        cfg,
        trainset,
        valset,
        loss_func_params=loss_func_params,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        image_dump_interval=1000,
        checkpoint_interval=200,
        validation_interval=1,
    )
    trainer.run(num_epochs=1001, validation=True)