import numpy as np
import SimpleITK as sitk


def get_mean_std_from_image(image):
    """ Get mean and standard deviation from the input image.
    """
    assert isinstance(image, sitk.Image)

    image_npy = sitk.GetArrayFromImage(image)
    return np.mean(image_npy), np.std(image_npy)


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


class FixedNormalizer(object):
  """
  Use fixed mean and stddev to normalize image intensities
  intensity = (intensity - mean) / stddev
  if clip is enabled:
      intensity = np.clip((intensity - mean) / stddev, -1, 1)
  """

  def __init__(self, mean, stddev, clip=True):
    """ constructor """
    assert stddev > 0, 'stddev must be positive'
    assert isinstance(clip, bool), 'clip must be a boolean'
    self.mean = mean
    self.stddev = stddev
    self.clip = clip

  def __call__(self, image):
    """ normalize image """
    if isinstance(image, sitk.Image):
      return normalize_image(image, self.mean, self.stddev, self.clip)

    elif isinstance(image, (list, tuple)):
      for idx, im in enumerate(image):
        assert isinstance(im, sitk.Image)
        image[idx] = normalize_image(im, self.mean, self.stddev, self.clip)
      return image

    else:
      raise ValueError('Unknown type of input. Normalizer only supports Image3d or Image3d list/tuple')

  def to_dict(self):
    """ convert parameters to dictionary """
    obj = {'type': 0, 'mean': self.mean, 'stddev': self.stddev, 'clip': self.clip}
    return obj


class AdaptiveNormalizer(object):
  """
  Normalize image using z-score normalization.
  """

  def __init__(self, clip_sigma=3):
    """
    :param clip_sigma: clip the intensity within the 'clip_sigma' standard deviation. 68% voxels lies within 1
      standard deviation, 95% within 2 standard deviation, and 99.7% within 3 standard deviation.
    """
    assert clip_sigma > 0
    self.clip_sigma = clip_sigma

  def normalize(self, single_image):
    """ Normalize a given image """
    assert isinstance(single_image, sitk.Image), 'image must be an image3d object'

    normalize_mean, normalize_stddev = get_mean_std_from_image(single_image)
    normalize_stddev = max(normalize_stddev, 1e-6)

    return normalize_image(single_image, normalize_mean, normalize_stddev, True, -self.clip_sigma, self.clip_sigma)

  def __call__(self, image):
    """ normalize image """
    if isinstance(image, sitk.Image):
      return self.normalize(image)

    elif isinstance(image, (list, tuple)):
      for idx, im in enumerate(image):
        assert isinstance(im, sitk.Image)
        image[idx] = self.normalize(im)
      return image

    else:
      raise ValueError('Unknown type of input. Normalizer only supports Image3d or Image3d list/tuple')

  def to_dict(self):
    """ convert parameters to dictionary """
    obj = {'type': 1, 'clip_sigma': self.clip_sigma}
    return obj