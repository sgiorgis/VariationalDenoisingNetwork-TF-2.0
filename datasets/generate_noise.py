import glob
import os
import cv2
import numpy as np
import h5py as h5

from math import floor
from arguments import parse_noise_generation_arguments

# chose this seen to match the original authors' work
np.random.seed(10000)


def peaks(n):
    """
       Implementation of the matlab function peaks
       Taken from: https://github.com/zsyOAOA/VDNet/blob/master/utils.py
       Parameters:
           n: The dimension of the original input
    """

    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    [xx, yy] = np.meshgrid(x, y)

    zz = 3 * (1 - xx) ** 2 * np.exp(-xx ** 2 - (yy + 1) ** 2) \
         - 10 * (xx / 5.0 - xx ** 3 - yy ** 5) * np.exp(-xx ** 2 - yy ** 2) \
         - 1 / 3.0 * np.exp(-(xx + 1) ** 2 - yy ** 2)

    return zz


def sincos_kernel():
    """
       Implementation of the sin cos kernel
       Taken from: https://github.com/zsyOAOA/VDNet/blob/master/utils.py
    """
    [xx, yy] = np.meshgrid(np.linspace(1, 10, 256), np.linspace(1, 20, 256))
    return np.sin(xx) + np.cos(yy)


def generate_gauss_kernel_mix(height, width):
    """
       Generate a H x W mixture Gaussian kernel with mean (center) and std (scale).
       Taken from: https://github.com/zsyOAOA/VDNet/blob/master/utils.py
       Parameters:
           height: The height dimension of original image
           width: The width dimension of original image
    """

    patch_size = 32

    k_height = floor(height / patch_size)
    k_width = floor(width / patch_size)

    height_indexes = np.arange(k_height) * patch_size
    width_indexes = np.arange(k_width) * patch_size

    center_width = np.random.uniform(low=0, high=patch_size, size=(k_height, k_width))
    center_width += width_indexes.reshape((1, -1))
    center_width = center_width.reshape((1, 1, k_height * k_width)).astype(np.float32)

    center_height = np.random.uniform(low=0, high=patch_size, size=(k_height, k_width))
    center_height += height_indexes.reshape((-1, 1))
    center_height = center_height.reshape((1, 1, k_height * k_width)).astype(np.float32)

    scale = np.random.uniform(low=patch_size / 2, high=patch_size, size=(1, 1, k_height * k_width))
    scale = scale.astype(np.float32)

    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    xx = xx[:, :, np.newaxis].astype(np.float32)
    yy = yy[:, :, np.newaxis].astype(np.float32)
    zz = 1. / (2 * np.pi * scale ** 2) * np.exp(
        (-(xx - center_width) ** 2 - (yy - center_height) ** 2) / (2 * scale ** 2))

    return zz.sum(axis=2, keepdims=False) / k_height * k_width


def generate_awgn(image, sigma):
    """
        Generates additive white gaussian noise and adds it on the image
        Parameters:
            image: The clean image to add noise on
            sigma: The noise level
        Returns:
            The Noisy image

    """
    height, width, channels = image.shape

    gauss = np.zeros((height, width, channels))

    for chn in range(channels):
        gauss[:, :, chn] = np.random.normal(0, sigma / 255, (height, width))

    noisy_image = image + gauss

    return noisy_image


def generate_iid_noise(data_directory, data_name, extension):
    """
        Generates white Gaussian noise for all image inputs for different noise levels
        Parameters:
            data_directory: The directory containing the images
            data_name: The dataset name (ex. LIVE1)
            extension: The extension of the images to search for
    """

    paths_images = glob.glob(os.path.join(os.path.join(data_directory, data_name), '**.' + extension),
                             recursive=True)

    sigmas = {
        'sigma15': 15,
        'sigma25': 25,
        'sigma50': 50
    }

    files = {
        'sigma15': h5.File(os.path.join(data_directory, data_name) + '_sigma15.hdf5', 'w'),
        'sigma25': h5.File(os.path.join(data_directory, data_name) + '_sigma25.hdf5', 'w'),
        'sigma50': h5.File(os.path.join(data_directory, data_name) + '_sigma50.hdf5', 'w')
    }

    clean = h5.File(os.path.join(data_directory, data_name) + '_clean.hdf5', 'w')

    for path_image in paths_images:

        image = cv2.imread(path_image, 1)[:, :, ::-1]

        clean.create_dataset(
            name=os.path.basename(path_image),
            dtype=image.dtype,
            shape=image.shape,
            data=image
        )

        for sigma_key in sigmas.keys():

            height, weight, channels = image.shape

            height -= int(height % 16)
            weight -= int(weight % 16)

            gauss = np.zeros((height, weight, channels))

            for chn in range(channels):
                gauss[:, :, chn] = np.random.normal(0, sigmas[sigma_key] / 255, (height, weight))

            files[sigma_key].create_dataset(
                name=os.path.basename(path_image),
                dtype=gauss.dtype,
                shape=gauss.shape,
                data=gauss
            )


def generate_noniid_noise(data_directory, data_name, extension):
    """
       Generates non-i.i.d noise for different kernels
       Parameters:
           data_directory: The directory containing the images
           data_name: The dataset name (ex. LIVE1)
           extension: The extension of the images to search for
    """
    paths_images = glob.glob(
        os.path.join(os.path.join(data_directory, data_name), '**.' + extension), recursive=True
    )

    sigma_max = 75 / 255.0
    sigma_min = 10 / 255.0

    files = {
        'case1': h5.File(os.path.join(data_directory, data_name) + '_case1.hdf5', 'w'),
        'case2': h5.File(os.path.join(data_directory, data_name) + '_case2.hdf5', 'w'),
        'case3': h5.File(os.path.join(data_directory, data_name) + '_case3.hdf5', 'w')
    }

    clean = h5.File(os.path.join(data_directory, data_name) + '_clean.hdf5', 'w')

    kernels = {
        'case1': peaks(256),
        'case2': sincos_kernel(),
        'case3': generate_gauss_kernel_mix(256, 256)
    }

    for path_image in paths_images:

        image = cv2.imread(path_image, 1)[:, :, ::-1]

        clean.create_dataset(
            name=os.path.basename(path_image),
            dtype=image.dtype,
            shape=image.shape,
            data=image
        )

        for kernel_key in kernels.keys():
            sigma = kernels[kernel_key]
            sigma = sigma_min + (sigma - sigma.min()) / (sigma.max() - sigma.min()) * (sigma_max - sigma_min)

            height, weight, channels = image.shape

            height -= int(height % 16)
            weight -= int(weight % 16)

            sigma = cv2.resize(sigma, (weight, height))
            sigma = sigma.astype(np.float32)

            noise = np.random.randn(height, weight, channels) * np.expand_dims(sigma, 2)
            noise = noise.astype(np.float32)

            data = np.concatenate((noise, sigma[:, :, np.newaxis]), axis=2)

            files[kernel_key].create_dataset(
                name=os.path.basename(path_image),
                dtype=data.dtype,
                shape=data.shape,
                data=data
            )


def main():
    args = parse_noise_generation_arguments()

    data_directory = args.data_directory
    data_name = args.data_name
    extension = args.extension

    if args.mode == 'non-iid':
        generate_noniid_noise(data_directory, data_name, extension)
    else:
        generate_iid_noise(data_directory, data_name, extension)


if __name__ == '__main__':
    main()
