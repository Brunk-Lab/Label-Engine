import argparse


def main():
    valset = ecDNADataset(
        dataset_path=cfg.ECDNA_PATH,
        split='val',
        augmentator=val_augmentator,
        points_sampler=points_sampler,
        with_image_info=True,
        celline='NCIH2170',
        date='0609_2024',
    )



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', type=str,
                        help='Path to the model script.')

    parser.add_argument('image_path', type=str,
                        help='Path to the image file/folder.')
    
    parser.add_argument('gpu_id', type=int,
                        help='GPU id. Set to -1 if using CPU.')
    
    

    return parser.parse_args()
