import argparse
import os
from pprint import pprint


def parse_validation_processing_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory', type=str, default='noisy_mat_file',
                        help="The noisy mat file to use for benchmark(default:non-iid)")
    parser.add_argument('--clean_mat_file', type=str, default='noisy_mat_file',
                        help="The noisy mat file to use for benchmark(default:non-iid)")
    parser.add_argument('--noisy_mat_file', type=str, default='noisy_mat_file',
                        help="The noisy mat file to use for benchmark(default:non-iid)")

    arguments = parser.parse_args()
    pprint(vars(arguments))

    return arguments


def parse_benchmark_processing_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--noisy_mat_file', type=str, default='noisy_mat_file',
                        help="The noisy mat file to use for benchmark(default:non-iid)")
    parser.add_argument('--checkpoint_directory', type=str, default='./checkpoints',
                        help="Checkpoints directory,  (default:./checkpoints)")

    arguments = parser.parse_args()
    pprint(vars(arguments))

    return arguments


def parse_noise_generation_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='non-iid',
                        help="Type of the noise to generate (default:non-iid)")
    parser.add_argument('--data_directory', type=str, default='./data',
                        help="Directory of the dataset to generate noise for, (default:./)")
    parser.add_argument('--data_name', type=str, default='./LIVE1',
                        help="Name of the dataset to generate noise for ex.LIVE1, (default:./LIVE1)")
    parser.add_argument('--extension', type=str, default='bmp',
                        help="Extension of the images to use as the dataset, (default:bmp)")

    arguments = parser.parse_args()
    pprint(vars(arguments))

    return arguments


def parse_validate_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--validation_clean_dataset_path', type=str, default='./',
                        help="Directory of clean validation dataset, (default:./)")
    parser.add_argument('--validation_noisy_dataset_path', type=str, default='./',
                        help="Directory of noisy validation dataset, (default:./)")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size of training, (default:64)")
    parser.add_argument('--checkpoint_directory', type=str, default='./checkpoints',
                        help="Checkpoints directory,  (default:./checkpoints)")
    arguments = parser.parse_args()
    pprint(vars(arguments))

    return arguments


def parse_test_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--validation_clean_dataset_path', type=str, default='./',
                        help="Directory of clean validation dataset, (default:./)")
    parser.add_argument('--validation_noisy_dataset_path', type=str, default='./',
                        help="Directory of noisy validation dataset, (default:./)")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size of training, (default:64)")
    parser.add_argument('--checkpoint_directory', type=str, default='./checkpoints',
                        help="Checkpoints directory,  (default:./checkpoints)")
    parser.add_argument('--save_directory', type=str, default='./images',
                        help="Saved images directory,  (default:./images)")
    parser.add_argument('--verbose', type=bool, default=False,
                        help="Verbose will save denoised images in --save_directory,  (default:False)")
    arguments = parser.parse_args()
    validate_test_arguments(arguments)
    pprint(vars(arguments))

    return arguments


def parse_train_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dataset_base_path', type=str, default='./',
                        help="Directory of train dataset, (default:./)")
    parser.add_argument('--validation_clean_dataset_path', type=str, default='./',
                        help="Directory of clean validation dataset, (default:./)")
    parser.add_argument('--validation_noisy_dataset_path', type=str, default='./',
                        help="Directory of noisy validation dataset, (default:./)")

    parser.add_argument('--batches', type=int, default=5000, help="Number of batches, (default:5000)")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size of training, (default:64)")
    parser.add_argument('--patch_size', type=int, default=128, help="Patch size of data sample,  (default:128)")

    parser.add_argument('--epochs', type=int, default=60, help="Training epochs,  (default:60)")
    parser.add_argument('--learning_rate', type=float, default=4e-4, help="Learning rate,  (default:4e-4)")
    parser.add_argument('--epsilon', type=float, default=1e-3, help="Epsilon,  (default:1e-3)")
    parser.add_argument('--radious', type=int, default=3, help="Learning rate,  (default:3)")

    parser.add_argument('--restore_model', type=bool, default=False, help="Restore loaded model,  (default:false)")
    parser.add_argument('--checkpoint_directory', type=str, default='./checkpoints',
                        help="Checkpoints directory,  (default:./checkpoints)")
    parser.add_argument('--checkpoint_frequency', type=int, default='5',
                        help="Frequency to take snapshots of the model,  (default:5)")

    arguments = parser.parse_args()
    validate_train_arguments(arguments)
    pprint(vars(arguments))

    return arguments


def validate_test_arguments(arguments):
    if not os.path.exists(arguments.checkpoint_directory):
        raise Exception("Checkpoint directory {} does not exist".format(arguments.checkpoint_directory))

    if not os.path.exists(arguments.validation_clean_dataset_path):
        raise Exception(
            "Validation clean dataset path {} does not exist".format(arguments.validation_clean_dataset_path))

    if not os.path.exists(arguments.validation_noisy_dataset_path):
        raise Exception(
            "Validation noisy dataset path {} does not exist".format(arguments.validation_noisy_dataset_path))


def validate_train_arguments(arguments):
    if not os.path.exists(arguments.train_dataset_base_path):
        raise Exception("Train dataset base path {} does not exist".format(arguments.train_dataset_base_path))

    if not os.path.exists(arguments.validation_clean_dataset_path):
        raise Exception(
            "Validation clean dataset path {} does not exist".format(arguments.validation_clean_dataset_path))

    if not os.path.exists(arguments.validation_noisy_dataset_path):
        raise Exception(
            "Validation noisy dataset path {} does not exist".format(arguments.validation_noisy_dataset_path))

    if not os.path.exists(arguments.checkpoint_directory):
        os.makedirs(arguments.checkpoint_directory)
