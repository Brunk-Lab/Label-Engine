import imageio
import numpy as np
import os
import SimpleITK as sitk
import torch


type_conversion_from_numpy_to_sitk = {
    np.int8:     sitk.sitkInt8,
    np.int16:    sitk.sitkInt16,
    np.int32:    sitk.sitkInt32,
    np.int32:      sitk.sitkInt32,
    np.int64:    sitk.sitkInt64,
    np.uint8:    sitk.sitkUInt8,
    np.uint16:   sitk.sitkUInt16,
    np.uint32:   sitk.sitkUInt32,
    np.uint64:   sitk.sitkUInt64,
    np.uint:     sitk.sitkUInt32,
    np.float32:  sitk.sitkFloat32,
    np.float64:  sitk.sitkFloat64,
    np.float32:    sitk.sitkFloat32
}


def get_image_frame(image):
    """
    Get the frame of the given image. An image frame contains the origin, spacing, and direction of a image.

    :parma image: a SimpleITK image
    :return frame: the frame packed in a numpy array
    """
    assert isinstance(image, sitk.Image)

    frame = []
    frame.extend(list(image.GetSpacing()))
    frame.extend(list(image.GetOrigin()))
    frame.extend(list(image.GetDirection()))

    return np.array(frame, dtype=np.float32)


def set_image_frame(image, frame):
    """
    Set the frame of the SimpleITK image

    :param image: the a new frame to the input image.
    :param frame: the new frame of the image. It is a numpy array with 15 elements, with the first three elements
                  representing the spacing, the next three elements representing the origin, and the rest representing
                  the direction.
    """
    assert isinstance(image, sitk.Image)

    spacing = frame[:2].astype(np.double)
    origin = frame[2:4].astype(np.double)
    direction = frame[4:8].astype(np.double)

    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)


def read_picture(picture_path, out_image_type=np.float32):
    """
    Convert picture to SimpleITK type
    """
    assert os.path.isfile(picture_path)

    image_array = imageio.imread(picture_path)
    image_npy = np.array(image_array, dtype=out_image_type)

    image = sitk.GetImageFromArray(image_npy)

    return image


def write_picture(image, save_path):
    """
    Write picture from SimpleITK image
    """
    image_npy = sitk.GetArrayFromImage(image)
    imageio.imwrite(save_path, image_npy)


def save_intermediate_results(idxs, crops, masks, outputs, frames, file_names, out_folder):
    """
    Save intermediate results to training folder

    :param idxs: the indices of crops within batch to save
    :param crops: the batch tensor of image crops
    :param masks: the batch tensor of segmentation crops
    :param outputs: the batch tensor of output label maps
    :param frames: the batch frames
    :param file_names: the batch file names
    :param out_folder: the batch output folder
    :return: None
    """
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    for i in idxs:

        case_out_folder = os.path.join(out_folder, file_names[i])
        if not os.path.isdir(case_out_folder):
            os.makedirs(case_out_folder)

        if crops is not None:
            images = convert_tensor_to_image(crops[i], dtype=np.float32)
            frame = frames[i].numpy()
            for modality_idx, image in enumerate(images):
                set_image_frame(image, frame)
                sitk.WriteImage(image, os.path.join(case_out_folder, 'batch_{}_crop_{}.nii.gz'.format(i, modality_idx)))

        if masks is not None:
            mask = convert_tensor_to_image(masks[i, 0], dtype=np.float32)
            set_image_frame(mask, frames[i].numpy())
            sitk.WriteImage(mask, os.path.join(case_out_folder, 'batch_{}_mask.nii.gz'.format(i)))

        if outputs is not None:
            cls_num = outputs.size()[1]
            for cls in range(cls_num):
                output = convert_tensor_to_image(outputs[i, cls].data, dtype=np.float32)
                set_image_frame(output, frames[i].numpy())
                sitk.WriteImage(output, os.path.join(case_out_folder, 'batch_{}_output_{}.nii.gz'.format(i, cls)))


def crop_image(image, cropping_center, cropping_size, cropping_spacing, interp_method, dim):
    """
    Crop a patch from a volume given the cropping center, cropping size, cropping spacing, and the interpolation method.
    This function DO NOT consider the transformation of coordinate systems, which means the cropped patch has the same
    coordinate system with the given volume.

    :param image: the given volume to be cropped.
    :param cropping_center: the center of the cropped patch in the world coordinate system of the given volume.
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

    cropping_physical_size = [cropping_size[idx] * cropping_spacing[idx] for idx in range(dim)]
    cropping_start_point_world = [cropping_center[idx] - cropping_physical_size[idx] / 2.0 for idx in range(dim)]
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
    outimage = sitk.Resample(image, cropping_size, transform, interp_method, cropping_origin, cropping_spacing,
                             cropping_direction)

    return outimage


def copy_image(source_image, target_start_voxel, target_end_voxel, target_image, dim=2):
    """
    Copy from the source image to the target image in a given rectangle region.
    The source image and the target image should have the same orientation.
    """
    assert isinstance(source_image, sitk.Image)
    assert isinstance(target_image, sitk.Image)

    for idx in range(dim):
        target_start_voxel[idx] = int(target_start_voxel[idx])
        target_end_voxel[idx] = int(target_end_voxel[idx])

    target_start_world = target_image.TransformIndexToPhysicalPoint(target_start_voxel)
    source_start_voxel = source_image.TransformPhysicalPointToIndex(target_start_world)
    paste_size = [int(target_end_voxel[idx] - target_start_voxel[idx]) for idx in range(dim)]

    return sitk.Paste(target_image, source_image, paste_size, source_start_voxel, target_start_voxel)


def image_partition_by_fixed_size(image, bbox_start_voxel, bbox_end_voxel,
                                  partition_size, partition_stride, max_stride):
    """
    Split image by fixed size.

    :param image: the input image to be spilt
    :param bbox_start_voxel: the partition start voxel (inclusive)
    :param bbox_end_voxel: the partition end voxel (exclusive)
    :param partition_size: the physical size of each partition
    :param partition_stride: the partition stride
    :return partition_centers: the list containing center voxel of each partition
    """
    image_size, image_spacing, image_origin = image.GetSize(), image.GetSpacing(), image.GetOrigin()
    for idx in range(2):
        assert image_size[idx] >= max_stride and image_size[idx] % max_stride == 0

    bbox_size = [min(image_size[idx], bbox_end_voxel[idx] - bbox_start_voxel[idx]) for idx in range(2)]
    for idx in range(2):
        if bbox_size[idx] % max_stride != 0:
            bbox_size[idx] = max_stride * (bbox_size[idx] // max_stride + 1)
        bbox_size[idx] = min(bbox_size[idx], image_size[idx])
        bbox_end_voxel[idx] = bbox_start_voxel[idx] + bbox_size[idx]
        if bbox_end_voxel[idx] > image_size[idx]:
            bbox_end_voxel[idx] = image_size[idx]
            bbox_start_voxel[idx] = bbox_end_voxel[idx] - bbox_size[idx]
        assert bbox_start_voxel[idx] >= 0

    box_size = [int(partition_size[idx] / image_spacing[idx] + 0.5) for idx in range(2)]
    for idx in range(2):
        if box_size[idx] % max_stride:
            box_size[idx] = max_stride * (box_size[idx] // max_stride + 1)
        box_size[idx] = min(bbox_size[idx], box_size[idx])

    stride_size = [int(partition_stride[idx] / image_spacing[idx] + 0.5) for idx in range(2)]
    for idx in range(2):
        stride_size[idx] = min(bbox_size[idx], stride_size[idx])

    num_partitions = [int(np.ceil((bbox_size[idx] - box_size[idx]) / stride_size[idx]) + 1) for idx in range(3)]
    start_voxels, end_voxels = [], []
    for idx in range(0, num_partitions[0]):
        for idy in range(0, num_partitions[1]):
            start_voxel = [bbox_start_voxel[0] + idx * stride_size[0],
                           bbox_start_voxel[1] + idy * stride_size[1]
                           ]
            end_voxel = [start_voxel[0] + box_size[0], start_voxel[1] + box_size[1]]
            for dim in range(2):
                if end_voxel[dim] > bbox_end_voxel[dim]:
                    end_voxel[dim] = bbox_end_voxel[dim]
                    start_voxel[dim] = end_voxel[dim] - box_size[dim]
                    assert start_voxel[dim] >= 0

            start_voxels.append(start_voxel)
            end_voxels.append(end_voxel)

    return start_voxels, end_voxels


def normalize_image(image, mean, std, clip, clip_min=-1.0, clip_max=1.0):
    """
    Normalize image by setting mean and standard deviation.
    """
    assert isinstance(image, sitk.Image)

    image_npy = sitk.GetArrayFromImage(image)
    image_npy = (image_npy - mean) / std

    if clip:
        image_npy[image_npy < clip_min] = clip_min
        image_npy[image_npy > clip_max] = clip_max

    normalized_image = sitk.GetImageFromArray(image_npy)
    set_image_frame(normalized_image, get_image_frame(image))
    normalized_image = sitk.Cast(normalized_image, image.GetPixelID())

    return normalized_image


def percentiles(image, percentiles):
    """
    Get image percentile
    """
    assert isinstance(image, sitk.Image)

    image_npy = sitk.GetArrayFromImage(image)
    results = np.percentile(image_npy, percentiles)
    return results


def select_random_voxels_in_multi_class_mask(mask, num_selected, selected_label):
    """
    Randomly select a list of voxels with the given label in the mask

    :param mask: A multi-class label image
    :param num_selected: the number of voxels to be selected
    :param selected_label: the label to which the selected voxels belong
    """
    assert isinstance(mask, sitk.Image)

    mask_npy = sitk.GetArrayFromImage(mask)
    valid_voxels = np.argwhere(mask_npy == selected_label)

    selected_voxels = []
    while len(valid_voxels) > 0 and len(selected_voxels) < num_selected:
        selected_index = np.random.randint(0, len(valid_voxels))
        selected_voxel = valid_voxels[selected_index]
        selected_voxels.append(selected_voxel[::-1])

    return selected_voxels


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


def convert_tensor_to_image(tensor, dtype):
    """
    convert tensor to SimpleITK image object
    """
    assert isinstance(tensor, torch.Tensor), 'input must be a tensor'

    data = tensor.cpu().numpy()

    if tensor.dim() == 2:
        # single channel 2d image volume
        image = sitk.GetImageFromArray(data)

        if dtype is not None and dtype in type_conversion_from_numpy_to_sitk.keys():
            sitk_type = type_conversion_from_numpy_to_sitk[dtype]
            image = sitk.Cast(image, sitk_type)

    elif tensor.dim() == 3:
        # multi-channel 2d image volume
        image = []
        for i in range(data.shape[0]):
            tmp = sitk.GetImageFromArray(data[i])

            if dtype is not None and dtype in type_conversion_from_numpy_to_sitk.keys():
                sitk_type = type_conversion_from_numpy_to_sitk[dtype]
                tmp = sitk.Cast(tmp, sitk_type)
            image.append(tmp)
    else:
        raise ValueError('Only supports 3-dimsional or 4-dimensional image volume')

    return image


def resample(image, reference, interp_method, dim=2):
    """ Resample image based on inference image
    """
    assert isinstance(image, sitk.Image)
    assert isinstance(reference, sitk.Image)

    if interp_method == 'LINEAR':
        interp_method = sitk.sitkLinear
    elif interp_method == 'NN':
        interp_method = sitk.sitkNearestNeighbor
    else:
        raise ValueError('Unsupported interpolation type.')

    identity_transform = sitk.Transform(dim, sitk.sitkIdentity)
    return sitk.Resample(image, reference, identity_transform, interp_method)


def resample_spacing(image, resampled_spacing, max_stride, interp_method, dim=2):
    """ Resample the spacing of image

    :param image: the input image.
    :param resampled_spacing: the spacing of the resampled output image.
    :param max_stride: the output size should be the multiple
    :param interp_method: the interpolation method.
    :return the resampled image
    """
    assert isinstance(image, sitk.Image)

    in_spacing = image.GetSpacing()
    in_size = image.GetSize()
    in_origin = [float(image.GetOrigin()[idx]) for idx in range(dim)]
    in_direction = [float(image.GetDirection()[idx]) for idx in range(dim * dim)]

    out_spacing = [float(resampled_spacing[idx]) for idx in range(dim)]
    out_size = [int(in_size[idx] * in_spacing[idx] / out_spacing[idx] + 0.5) for idx in range(dim)]
    for idx in range(dim):
        if out_size[idx] % max_stride:
            out_size[idx] = max_stride * (out_size[idx] // max_stride + 1)

    if interp_method == 'LINEAR':
        interp_method = sitk.sitkLinear
    elif interp_method == 'NN':
        interp_method = sitk.sitkNearestNeighbor
    else:
        raise ValueError('Unsupported interpolation type.')

    identity_transform = sitk.Transform(dim, sitk.sitkIdentity)
    return sitk.Resample(image, out_size, identity_transform, interp_method, in_origin, resampled_spacing,
                         in_direction)


def get_num_connected_component(mask, threshold):
    """ Return the number of connected components
    :param mask: a binary mask
    :param threshold: the threshold below which the voxel will be set as 0.
    """
    assert isinstance(mask, sitk.Image)

    filter = sitk.ConnectedComponentImageFilter()
    filter.SetFullyConnected(True)

    mask_cc = filter.Execute(mask)
    mask_binary_cc = sitk.RelabelComponent(mask_cc, threshold)
    max_val = np.max(sitk.GetArrayFromImage(mask_binary_cc))

    return max_val


def pick_largest_connected_component(mask, labels):
    """ Pick the largest connected component.
    :param mask: The multi label mask
    :param labels: The labels which will be selected the largest connect component.
    """
    assert isinstance(mask, sitk.Image)
    assert isinstance(labels, list)

    filter = sitk.ConnectedComponentImageFilter()
    filter.SetFullyConnected(True)

    largest_cc_binaries = []
    for label in labels:
      mask_binary = (mask == label)
      mask_binary_cc = filter.Execute(mask_binary)

      mask_binary_cc = sitk.RelabelComponent(mask_binary_cc)
      mask_binary_largest_cc = (mask_binary_cc == 1)
      largest_cc_binaries.append(mask_binary_largest_cc)

    largest_cc_multi = largest_cc_binaries[0]
    for idx in range(1, len(labels)):
      largest_cc_multi = sitk.Add(largest_cc_multi, labels[idx] * largest_cc_binaries[idx])

    return sitk.Cast(largest_cc_multi, mask.GetPixelID())


def remove_small_connected_component(mask, labels, threshold):
    """ Pick the largest connected component.
    :param mask: the multi label mask
    :param labels: the labels which will be selected the largest connect component.
    :param threshold: the threshold below which the voxel will be set as 0.
    """
    assert isinstance(mask, sitk.Image)
    assert isinstance(labels, list)

    filter = sitk.ConnectedComponentImageFilter()
    filter.SetFullyConnected(True)

    largest_cc_binaries = []
    for label in labels:
      mask_binary = (mask == label)
      mask_binary_cc = filter.Execute(mask_binary)

      mask_binary_cc = sitk.RelabelComponent(mask_binary_cc, threshold)
      mask_binary_largest_cc = (mask_binary_cc > 0)
      largest_cc_binaries.append(mask_binary_largest_cc)

    largest_cc_multi = largest_cc_binaries[0]
    for idx in range(1, len(labels)):
      largest_cc_multi = sitk.Add(largest_cc_multi, labels[idx] * largest_cc_binaries[idx])

    return sitk.Cast(largest_cc_multi, mask.GetPixelID())


def add_image_value(image, start_voxel, end_voxel, value):
    """ Add value to the input image in a given volume.
    """
    assert isinstance(image, sitk.Image)

    for idx in range(2):
        start_voxel[idx] = int(start_voxel[idx])
        end_voxel[idx] = int(end_voxel[idx])

    image_npy = sitk.GetArrayFromImage(image)
    image_npy[start_voxel[1]:end_voxel[1], start_voxel[0]:end_voxel[0]] += value
    added_image = sitk.GetImageFromArray(image_npy)
    added_image.CopyInformation(image)

    return added_image


def get_mean_std_from_image(image):
    """ Get mean and standard deviation from the input image.
    """
    assert isinstance(image, sitk.Image)

    image_npy = sitk.GetArrayFromImage(image)
    return np.mean(image_npy), np.std(image_npy)


def get_bounding_box(mask, selected_labels):
    """ Get the bounding box of the image volume in the given intensity (inclusive).

    :param mask: multi-label mask
    :param selected_labels: a list containing all labels to get the bounding box
    """
    assert isinstance(mask, sitk.Image)

    mask_npy = sitk.GetArrayFromImage(mask)
    selected_mask_npy = np.zeros_like(mask_npy)
    if selected_labels is not None:
        for label in selected_labels:
            selected_mask_npy[mask_npy == label] = 1
    else:
        selected_mask_npy[mask_npy > 0] = 1

    selected_mask = sitk.GetImageFromArray(selected_mask_npy)
    selected_mask.CopyInformation(mask)

    bbox_filter = sitk.LabelShapeStatisticsImageFilter()
    try:
        bbox_filter.Execute(selected_mask)
        bbox = np.array(bbox_filter.GetBoundingBox(1))
        bbox_start_voxel = [bbox[0], bbox[1]]
        bbox_end_voxel = [bbox_start_voxel[0] + bbox[2], bbox_start_voxel[1] + bbox[3]]
    except:
        print('Fail to get the bounding box.')
        bbox_start_voxel, bbox_end_voxel = None, None

    return bbox_start_voxel, bbox_end_voxel