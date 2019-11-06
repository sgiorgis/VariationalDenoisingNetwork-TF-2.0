import cv2
import tensorflow as tf

from datasets.data_transformations import dataset_transformation
from datasets.datasets_generators.train_dataset_generator import TrainDatasetGenerator
from datasets.datasets_generators.validation_dataset_generator import ValidationDatasetGenerator
from datasets.datasets_generators.simulate_dataset_generator import SimulateTrainGenerator


def load_simulation_data(simulation_dataset_path, data_length, patch_size, radious, epsilon):
    """
       Load the train data for simulation
       Parameters:
           simulation_dataset_path: The path of the clean images to be used for training
           data_length: The number of data points to be used per epoch
           patch_size: The patch size to crop from the original images
           radious: The radious for the sigma estimate
           epsilon: The epsilon for the sigma estimate
    """
    # Load images in memory for performance issues
    images = [cv2.imread(image_path, 1)[:, :, ::-1] for image_path in simulation_dataset_path]

    dataset = generate_simulation_dataset(images, data_length, patch_size=patch_size, radious=radious, epsilon=epsilon)
    dataset = dataset.map(dataset_transformation, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset


def load_train_data(train_dataset_path, train_noisy_dataset_path, data_length, patch_size, radious, epsilon):
    """
       Load the train data for benchmark
       Parameters:
           train_dataset_path: The path of the clean images to be used for training
           train_noisy_dataset_path: The path of the noisy images to be used for training
           data_length: The number of data points to be used per epoch
           patch_size: The patch size to crop from the original images
           radious: The radious for the sigma estimate
           epsilon: The epsilon for the sigma estimate
    """

    # Load images in memory for performance issues
    images = [cv2.imread(image_path)[:, :, ::-1] for image_path in train_dataset_path]
    noisy_images = [cv2.imread(noisy_image_path)[:, :, ::-1] for noisy_image_path in train_noisy_dataset_path]

    dataset = generate_train_dataset(images, noisy_images, data_length, patch_size, radious, epsilon)
    dataset = dataset.map(dataset_transformation, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset


def load_validation_dataset(validation_clean_dataset_path, validation_noisy_dataset_path):
    """
       Load the validation data for benchmark
       Parameters:
           validation_clean_dataset_path: The path of the clean images to be used for validation
           validation_noisy_dataset_path: The path of the noisy images to be used for validation
    """

    validation_dataset = generate_validation_dataset(validation_clean_dataset_path, validation_noisy_dataset_path)
    validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return validation_dataset


def generate_validation_dataset(validation_clean_dataset_path, validation_noisy_dataset_path):
    """
       Generate custom tensorflow dataset for validation data for benchmark
       Parameters:
           validation_clean_dataset_path: The path of the clean images to be used for validation
           validation_noisy_dataset_path: The path of the noisy images to be used for validation
    """
    output_types = (tf.float32, tf.float32)
    output_shapes = (
        tf.TensorShape([256, 256, 3]),
        tf.TensorShape([256, 256, 3]),
    )

    return tf.data.Dataset.from_generator(
        ValidationDatasetGenerator(validation_clean_dataset_path, validation_noisy_dataset_path),
        output_types,
        output_shapes
    )


def generate_train_dataset(images, noisy_images, dataset_length, patch_size, radious, epsilon):
    """
       Generate custom tensorflow dataset for train data for benchmark
       Parameters:
           images: The path of the clean images to be used for validation
           noisy_images: The path of the noisy images to be used for validation
           dataset_length: The number of data points to be used per epoch
           patch_size: The patch size to crop from the original images
           radious: The radious for the sigma estimate
           epsilon: The epsilon for the sigma estimate
    """
    output_types = (tf.float32, tf.float32, tf.float32, tf.float32)
    output_shapes = (
        tf.TensorShape([128, 128, 3]),
        tf.TensorShape([128, 128, 3]),
        tf.TensorShape([128, 128, 3]),
        tf.TensorShape([1, 1, 1])
    )

    return tf.data.Dataset.from_generator(
        TrainDatasetGenerator(images, noisy_images, dataset_length, patch_size, radious, epsilon),
        output_types,
        output_shapes,
    )


def generate_simulation_dataset(images, dataset_length, patch_size, radious, epsilon):
    """
       Generate custom tensorflow dataset for train data for simulation
       Parameters:
           images: The path of the clean images to be used for validation
           dataset_length: The number of data points to be used per epoch
           patch_size: The patch size to crop from the original images
           radious: The radious for the sigma estimate
           epsilon: The epsilon for the sigma estimate
    """
    output_types = (tf.float32, tf.float32, tf.float32, tf.float32)
    output_shapes = (
        tf.TensorShape([128, 128, 3]),
        tf.TensorShape([128, 128, 3]),
        tf.TensorShape([128, 128, 3]),
        tf.TensorShape([1, 1, 1])
    )
    return tf.data.Dataset.from_generator(
        SimulateTrainGenerator(images, dataset_length, patch_size=patch_size, radious=radious, epsilon=epsilon),
        output_types,
        output_shapes,
    )
