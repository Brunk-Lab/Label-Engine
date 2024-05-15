import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', type=str,
                        help='Path to the model script.')

    parser.add_argument('image_path', type=str,
                        help='Path to the image file/folder.')
    
    parser.add_argument('gpu_id', type=int,
                        help='GPU id. Set to -1 if using CPU.')
    
    

    return parser.parse_args()
