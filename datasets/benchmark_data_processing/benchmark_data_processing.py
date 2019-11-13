import os
import sys

sys.path.append(os.getcwd())

import h5py as h5

from scipy.io import loadmat
from tqdm import tqdm

from arguments import parse_benchmark_processing_arguments


def save_hdf5(image, name, hdf5_file):
    hdf5_file.create_dataset(
        name=name,
        shape=image.shape,
        dtype=image.dtype,
        data=image
    )


def main():
    args = parse_benchmark_processing_arguments()
    noisy_data_mat_file = args.noisy_mat_file
    noisy_data_mat_name = os.path.basename(noisy_data_mat_file).replace('.mat', '')
    noisy_data_mat = loadmat(noisy_data_mat_file)[noisy_data_mat_name]
    noisy_data_hdf5_file = h5.File(noisy_data_mat_file.replace('mat', 'hdf5'), 'w')

    num_patch = 0

    for image_index in tqdm(range(noisy_data_mat.shape[0])):
        for block_index in range(noisy_data_mat.shape[1]):
            noisy_image = noisy_data_mat[image_index, block_index, :, :, :]
            save_hdf5(noisy_image, str(num_patch), noisy_data_hdf5_file)

            num_patch += 1


if __name__ == '__main__':
    main()
