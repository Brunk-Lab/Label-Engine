from pathlib import Path
import imageio
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


def convert_image_to_tensor(image):
    """
    Convert an SimpleITK image object to float tensor
    """
    if isinstance(image, sitk.Image):
        tensor = torch.from_numpy(sitk.GetArrayFromImage(image))
        tensor = torch.unsqueeze(tensor, 0)
        tensor = tensor.float()
    elif isinstance(image, list):
        tensor = []
        for i in range(len(image)):
            assert isinstance(image[i], sitk.Image)
            tmp = torch.from_numpy(sitk.GetArrayFromImage(image[i]))
            tmp = torch.unsqueeze(tmp, 0)
            tmp = tmp.float()
            tensor.append(tmp)
        tensor = torch.cat(tensor, 0)
    else:
        raise ValueError('unknown input type')

    return tensor


def crop_image(
        image, 
        cropping_center, 
        cropping_size, 
        cropping_spacing, 
        interp_method, 
        dim
    ):
    """
    Crop a patch from a volume. This function DOES NOT consider the transformation
    of coordinate systems, which means the cropped patch has the same coordinate 
    system with the given volume.

    :param image: the given volume to be cropped.
    :param cropping_center: the center of the cropped patch in the world coordinate 
                            system of the given volume.
    :param cropping_size: the voxel coordinate size of the cropped patch.
    :param cropping_spacing: the voxel spacing of the cropped patch.
    :param interp_method: the interpolation method, only support 'NN' and 'Linear'.
    :param dim: dimension of the input image
    :return a cropped patch
    """
    assert isinstance(image, sitk.Image)

    cropping_center = [float(cropping_center[idx]) for idx in range(dim)]
    cropping_size = [int(cropping_size[idx]) for idx in range(dim)]
    cropping_spacing = [float(cropping_spacing[idx]) for idx in range(dim)]

    cropping_physical_size = [cropping_size[idx] * cropping_spacing[idx] 
                              for idx in range(dim)]
    cropping_start_point_world = [
        cropping_center[idx] - cropping_physical_size[idx] / 2.0 for idx in range(dim)]
    for idx in range(dim):
        cropping_start_point_world[idx] += cropping_spacing[idx] / 2.0

    cropping_origin = cropping_start_point_world
    cropping_direction = image.GetDirection()

    if interp_method == 'LINEAR':
        interp_method = sitk.sitkLinear
    elif interp_method == 'NN':
        interp_method = sitk.sitkNearestNeighbor
    else:
        raise ValueError('Unsupported interpolation type.')

    transform = sitk.Transform(dim, sitk.sitkIdentity)
    outimage = sitk.Resample(
        image, cropping_size, transform, interp_method, cropping_origin, 
        cropping_spacing, cropping_direction
    )

    return outimage


def read_picture(picture_path, out_image_type=np.float32):
    """
    Convert picture to SimpleITK type
    """
    assert os.path.isfile(picture_path)

    image_array = imageio.imread(picture_path)
    image_npy = np.array(image_array, dtype=out_image_type)

    image = sitk.GetImageFromArray(image_npy)

    return image


class ecDNADataset(Dataset):
    def __init__(
        self,
        dataset_path,
        split='train',
        crop_size=[2048, 2448],
        augmentation_params=None,
        normalizer=None, 
    ):
        """ constructor
        :param dataset_path: the folder contains datasets
        :param crop_size: crop size, e.g., [2048, 2448]
        :param augmentation_params: a dictionary containing augmentation parameters
        :param normalizers: used to normalize the image crops, one for one image modality
        """
        dataset_path = Path(dataset_path)
        self.images_path = dataset_path / 'images'
        self.masks_path = dataset_path / 'masks'
        self.coords_path = dataset_path / 'coords'

        self.split = split
        if split == 'train':
            imlist_file = dataset_path / 'datasets' / 'train_0422_2024.txt'
        elif split == 'val':
            imlist_file = dataset_path / 'datasets' / 'val_0422_2024.txt'
        else:
            raise ValueError(f'The split type: {split} is not supported.')

        self.im_name_list = []
        self.im_path_list = []
        self.im_mask_list = []
        self.im_coords_list = []
        f = open(imlist_file, 'r')
        for line in f:
            im_name = line.split('.')[0]
            im_path = str(self.images_path / line.strip())
            mask_path = str(self.masks_path / line.strip())
            coords_path = str(self.coords_path / f'{im_name}.npy')
            self.im_name_list.append(im_name)
            self.im_path_list.append(im_path)
            self.im_mask_list.append(mask_path)
            self.im_coords_list.append(coords_path)

        self.spacing = np.array([1.0, 1.0])
        self.crop_size = np.array(crop_size, dtype=np.int32)
        self.normalizer = normalizer

        if self.split == 'train':
            aug_params = augmentation_params
            self.random_translation = np.array(aug_params['random_translation'])
            self.random_scale = np.array(aug_params['random_scale'])
            self.random_vert_flip = aug_params['random_vert_flip']
            self.random_hori_flip = aug_params['random_hori_flip']

    def __len__(self):
        """ get the number of images in this data set """
        return len(self.im_path_list)

    def center_sample(self, image):
        """ return the world coordinate of the image center
        :param image: a image object
        :return: the image center in world coordinate
        """
        assert isinstance(image, sitk.Image)

        origin = image.GetOrigin()
        end_point_voxel = [int(image.GetSize()[idx] - 1) for idx in range(2)]
        end_point_world = image.TransformIndexToPhysicalPoint(end_point_voxel)

        center = np.array([(origin[idx] + end_point_world[idx]) / 2.0 \
                           for idx in range(2)], dtype=np.double)
        return center

    def __getitem__(self, index):
        """ get a training sample - image(s) and segmentation pair
        :param index:  the sample index
        :return cropped image, cropped mask, crop frame, case name
        """
        image_name = self.im_name_list[index]
        image_path = self.im_path_list[index]
        mask_path = self.im_mask_list[index]
        coords_path = self.im_coords_list[index]

        # image IO
        image = read_picture(image_path, np.float32)
        mask = read_picture(mask_path, np.float32)
        num_coords = len(np.load(coords_path))

        # foreground: 255, background: 0, invalid: other values
        mask_npy = sitk.GetArrayFromImage(mask)
        reordered_mask_npy = np.zeros_like(mask_npy) - 1        
        reordered_mask_npy[mask_npy == 0] = 0
        reordered_mask_npy[mask_npy == 255] = 1

        reordered_mask = sitk.GetImageFromArray(reordered_mask_npy)
        reordered_mask.CopyInformation(mask)
        mask = reordered_mask

        if self.split == 'train':
            # random translation
            center = self.center_sample(mask)
            center += np.random.uniform(-self.random_translation, self.random_translation, size=[2])

            # random rescale
            crop_spacing = self.spacing * np.random.uniform(self.random_scale[0], self.random_scale[1])

            # crop image and mask 
            image = crop_image(image, center, self.crop_size, crop_spacing, 'LINEAR', 2)
            mask = crop_image(mask, center, self.crop_size, crop_spacing, 'NN', 2)
            if self.normalizer is not None:
                image = self.normalizer(image)

            # random horizontal flip
            if self.random_hori_flip and np.random.random() > 0.5:
                image = sitk.Flip(image, [True, False], True)
                mask = sitk.Flip(mask, [True, False], True)

            # random vertical flip
            if self.random_vert_flip and np.random.random() > 0.5:
                image = sitk.Flip(image, [False, True], True)
                mask = sitk.Flip(mask, [False, True], True)

        if self.normalizer is not None:
            image = self.normalizer(image)

        # convert to tensors
        image = convert_image_to_tensor(image)
        mask = convert_image_to_tensor(mask)
        info = {'name': image_name, 'num_coords': num_coords}

        return image, mask, info