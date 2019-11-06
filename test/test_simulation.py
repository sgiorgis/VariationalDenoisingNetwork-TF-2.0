import os

import h5py
import tensorflow as tf
import numpy as np
from skimage import img_as_float, img_as_ubyte
from matplotlib import pyplot as plt

from tqdm import tqdm
from arguments import parse_test_arguments
from losses.loss import mse_function, ssim_function, psnr_function
from tensorflow.keras.metrics import Mean
from models.model_loader import load_test_model


def save_image(key, clean_image, noisy_image, denoised_image, save_directory):
    """
        Plots the three images clean, noisy and denoised and saves them to disk.
        Every image will get squeezed first to remove the first batch dimension
        and then is converted to 8-bit unsigned integer format.
        Parameters:
           key: The key describes the name of the images to save
           clean_image: The clean image to floating point format
           noisy_image: The noisy image to floating point format.
           denoised_image: The denoised image to floating point format.
           save_directory: The directory used to save the images
    """
    denoised_image = np.transpose(denoised_image.numpy().squeeze(), (0, 1, 2))
    denoised_image = img_as_ubyte(denoised_image.clip(0, 1))

    noisy_image = np.transpose(noisy_image.numpy().squeeze(), (0, 1, 2))
    noisy_image = img_as_ubyte(noisy_image.clip(0, 1))

    clean_image = np.transpose(clean_image.numpy().squeeze(), (0, 1, 2))
    clean_image = img_as_ubyte(clean_image)

    plt.subplot(131)
    plt.imshow(clean_image)
    plt.title('Clean image')
    plt.subplot(132)
    plt.imshow(noisy_image)
    plt.title('Noisy Image')
    plt.subplot(133)
    plt.imshow(denoised_image)
    plt.title('Denoised Image')

    plt.savefig(os.path.join(save_directory, key.replace('bmp', 'png')))


def validation_step(model, clean_image, noisy_image, losses):
    """
        Performs the validation step for every noisy image.
        Performs a forward pass to the model clips the denoised images
        and updates the loss functions.
        Parameters:
           model: The model to use for forward pass
           clean_image: The clean image to floating point format
           noisy_image: The noisy image to floating point format.
           losses: The mean value of all losses used for validation
    """
    dnet_output, snet_output = model(noisy_image)

    # clip the value so the values match the input (0,1)
    denoised_image = tf.clip_by_value(noisy_image - dnet_output[:, :, :, :3], 0, 1)

    losses['validation_mse'](mse_function(denoised_image, clean_image))
    losses['validation_ssim'](ssim_function(denoised_image, clean_image))
    losses['validation_psnr'](psnr_function(denoised_image, clean_image))

    return denoised_image


def test():
    device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'

    with tf.device(device):
        args = parse_test_arguments()

        losses = {
            'validation_mse': Mean(name='validation_mse'),
            'validation_psnr': Mean(name='validation_psnr'),
            'validation_ssim': Mean(name='validation_ssim')
        }

        model = load_test_model(args.checkpoint_directory)

        images = h5py.File(args.validation_clean_dataset_path, 'r')
        noises = h5py.File(args.validation_noisy_dataset_path, 'r')
        keys = images.keys()

        for key in tqdm(keys):
            clean_image = images[key]
            noise = noises[key][:, :, :3]

            height, width, _ = noise.shape
            clean_image = img_as_float(clean_image[:height, :width])
            noisy_image = clean_image + noise

            clean_image = tf.convert_to_tensor(clean_image[np.newaxis, :, :, :], dtype=tf.float32)
            noisy_image = tf.convert_to_tensor(noisy_image[np.newaxis, :, :, :], dtype=tf.float32)

            denoised_image = validation_step(model, clean_image, noisy_image, losses)

            if args.verbose:
                save_image(key, clean_image, noisy_image, denoised_image, args.save_directory)

        for loss in losses.keys():
            print("{}: {}".format(loss, losses[loss].result().numpy()))


if __name__ == '__main__':
    test()
