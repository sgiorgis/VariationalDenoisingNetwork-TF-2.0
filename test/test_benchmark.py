import os
import numpy as np
import tensorflow as tf
from skimage import img_as_float

from tqdm import tqdm
from scipy.io import loadmat, savemat
from arguments import parse_benchmark_processing_arguments
from models.model_loader import load_test_model


@tf.function
def denoise(model, noisy_image):
    dnet_output, snet_output = model(noisy_image)
    denoised_image = tf.clip_by_value(noisy_image - dnet_output[:, :, :, :3], 0, 1)

    return tf.image.convert_image_dtype(denoised_image, dtype=tf.uint8)


def test():
    device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'
    args = parse_benchmark_processing_arguments()

    with tf.device(device):

        model = load_test_model(args.checkpoint_directory)

        noisy_data_mat_file = args.noisy_mat_file
        noisy_data_mat_name = os.path.basename(noisy_data_mat_file).replace('.mat', '')
        noisy_data_mat = loadmat(noisy_data_mat_file)[noisy_data_mat_name]

        denoised_data = np.zeros(noisy_data_mat.shape, dtype=np.uint8)

        for image_index in tqdm(range(noisy_data_mat.shape[0])):
            for block_index in range(noisy_data_mat.shape[1]):
                noisy_image = noisy_data_mat[image_index, block_index, :, :, :]
                noisy_image = img_as_float(noisy_image[np.newaxis, :, :, :])
                noisy_image = tf.convert_to_tensor(noisy_image, dtype=tf.float32)

                denoised_data[image_index][block_index] = denoise(model, noisy_image).numpy()

        submit_data = {
            'SubmitSrgb': denoised_data
        }

        savemat(
            os.path.join(os.path.dirname(noisy_data_mat_file), 'SubmitSrgb.mat'),
            submit_data
        )


if __name__ == '__main__':
    test()
