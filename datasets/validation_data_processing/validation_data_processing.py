import os
import h5py as h5

from scipy.io import loadmat
from tqdm import tqdm
from arguments import parse_validation_processing_arguments


def save_hdf5(image, name, hdf5_file):
    hdf5_file.create_dataset(name=name, shape=image.shape, dtype=image.dtype, data=image)


def main():
    args = parse_validation_processing_arguments()
    data_directory = args.data_directory

    clean_data_mat_file = os.path.join(data_directory, args.clean_mat_file)
    noisy_data_mat_file = os.path.join(data_directory, args.noisy_mat_file)

    clean_data_mat_name = args.clean_mat_file.replace('.mat', '')
    noisy_data_mat_name = args.noisy_mat_file.replace('.mat', '')

    clean_data_mat = loadmat(clean_data_mat_file)[clean_data_mat_name]
    noisy_data_mat = loadmat(noisy_data_mat_file)[noisy_data_mat_name]

    clean_data_hdf5_file = h5.File(clean_data_mat_file.replace('mat', 'hdf5'), 'w')
    noisy_data_hdf5_file = h5.File(noisy_data_mat_file.replace('mat', 'hdf5'), 'w')

    num_patch = 0

    for image_index in tqdm(range(clean_data_mat.shape[0])):
        for block_index in range(clean_data_mat.shape[1]):
            noisy_image = noisy_data_mat[image_index, block_index, :, :, :]
            clean_image = clean_data_mat[image_index, block_index, :, :, :]
            save_hdf5(clean_image, str(num_patch), clean_data_hdf5_file)
            save_hdf5(noisy_image, str(num_patch), noisy_data_hdf5_file)

            num_patch += 1


if __name__ == '__main__':
    main()
