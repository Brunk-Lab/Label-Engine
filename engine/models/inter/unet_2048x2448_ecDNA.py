from core.utils.exp_imports.default import *

MODEL_NAME = 'inter_unet_2048x2448_ecDNA'


def main(cfg: Dict) -> None:
    model = build_model(img_size=(2048, 2448))
    train(model, cfg)


def build_model(img_size: Tuple[int, int]) -> interUNet:
    
    encoder_params = {'in_channels': 3, 'out_channels': 256}
    decoder_params = {'in_channels': 256, 'out_channels': 2}
    model = interUNet(
        encoder_params, 
        decoder_params,
        use_disks=True,
        auto=True,
    )
    max_stride = 16
    assert img_size[0] % max_stride == 0 and img_size[1] % max_stride == 0
    return model


def train(model: interUNet, cfg: Dict) -> None:
    cfg.img_size = (2048, 2448)
    cfg.num_max_points = 24
    cfg.num_max_next_points = 3
    cfg.seed = 0

    train_augmentator = Compose([
        Flip(),
        ShiftScaleRotate(
            shift_limit=0.03, 
            scale_limit=0,
            rotate_limit=(-3, 3), 
            border_mode=0, 
            p=0.75
        ),
        RandomBrightnessContrast(
            brightness_limit=(-0.25, 0.25),
            contrast_limit=(-0.15, 0.4), 
            p=0.75
        ),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
    ], p=1.0)

    val_augmentator = Compose([
        ResizeLongestSide(target_length=max(cfg.img_size)),
        PadIfNeeded(
            min_height=min(cfg.img_size), 
            min_width=min(cfg.img_size), 
            border_mode=0,
            position='top_left',
        ),
    ], p=1.0)

    points_sampler = MultiPointSampler(
        cfg.num_max_points, 
        prob_gamma=0.80,
        merge_objects_prob=0.15,
        max_num_merged_objects=2
    )

    trainset = ecDNADataset(
        dataset_path=cfg.ECDNA_PATH,
        split='train',
        augmentator=train_augmentator,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        with_image_info=True,
        date='0702_2024',
    )

    valset = ecDNADataset(
        dataset_path=cfg.ECDNA_PATH,
        split='val',
        augmentator=val_augmentator,
        points_sampler=points_sampler,
        with_image_info=True,
        date='0702_2024',
        celline='NCIH2170',
    )

    loss_func_params = {'name':'Focal', 'alpha':(0.5, 0.5), 'class_num':2}
    optimizer_params = {'name':'adam', 'lr':5e-5, 'betas':(0.9, 0.999), 'eps':1e-8}
    scheduler_params = {'name':'MultiStepLR', 'milestones':[1000, 1500], 'gamma':0.2}

    trainer = InterTrainer(
        model,
        cfg,
        trainset,
        valset,
        loss_func_params=loss_func_params,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        image_dump_interval=1,
        checkpoint_interval=10,
        validation_interval=10,
        seed=cfg.seed,
    )
    trainer.run(num_epochs=1, validation=True)