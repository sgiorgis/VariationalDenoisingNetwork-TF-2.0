import random
import tensorflow as tf


@tf.function
def dataset_normalize(noisy_image, clean_image, sigma, epsilon):
    """
        Normalizes the train tensors to float32 type scaling their values if needed.
        Args:
            noisy_image: The noisy image tensor to normalize
            clean_image: The clean image tensor to normalize
            sigma: The sigma map tensor to normalize
            epsilon: The epsilon that will not be normalize
        Returns:
            The normalized tensors
    """
    return tf.image.convert_image_dtype(noisy_image, dtype=tf.float32), \
           tf.image.convert_image_dtype(clean_image, dtype=tf.float32), sigma, epsilon


@tf.function
def dataset_validation_normalize(noisy_image, clean_image):
    """
        Normalizes the validation tensors to float32 type scaling their values if needed.
        Args:
            noisy_image: The noisy image tensor to normalize
            clean_image: The clean image tensor to normalize
        Returns:
            The normalized tensors
    """
    return tf.image.convert_image_dtype(noisy_image, dtype=tf.float32), \
           tf.image.convert_image_dtype(clean_image, dtype=tf.float32)


@tf.function
def dataset_transformation(noisy_image, clean_image, sigma, epsilon):
    """
        Transforms the tensors with one of the available transformations for 50% of the data.
        All noisy_image, clean_image and sigma map should be transformed together.
        Args:
            noisy_image: The noisy image tensor to transform
            clean_image: The clean image tensor to transform
            sigma: The sigma map tensor to transform
            epsilon: The epsilon that will not be transformed
        Returns:
            The rotated tensors
    """
    if random.randint(0, 1) == 1:
        return noisy_image, clean_image, sigma, epsilon

    transformations = [
        flip_up_down,
        rotate90,
        rotate90_up_down,
        rotate180,
        rotate180_up_down,
        rotate270,
        rotate270_up_down
    ]

    transformation = random.choice(transformations)

    noisy_image = transformation(noisy_image)
    clean_image = transformation(clean_image)
    sigma = transformation(sigma)

    return noisy_image, clean_image, sigma, epsilon


@tf.function
def rotate90_up_down(x):
    """
        Rotate the tensor 90 degrees and then flips it up down
        Args:
            x: The tensor to rotate and flip
        Returns:
            The rotated tensor
    """
    return flip_up_down(rotate90(x))


@tf.function
def rotate180_up_down(x):
    """
        Rotate the tensor 180 degrees and then flips it up down
        Args:
            x: The tensor to rotate and flip
        Returns:
            The rotated tensor
    """
    return flip_up_down(rotate180(x))


@tf.function
def rotate270_up_down(x):
    """
        Rotate the tensor 270 degrees and then flips it up down
        Args:
            x: The tensor to rotate and flip
        Returns:
            The rotated tensor
    """
    return flip_up_down(rotate180(x))


@tf.function
def rotate90(x: tf.Tensor) -> tf.Tensor:
    """
        Rotate the tensor 90 degrees
        Args:
            x: The tensor to rotate
        Returns:
            The rotated tensor
    """
    return tf.image.rot90(x)


@tf.function
def rotate180(x):
    """
        Rotate the tensor 180 degrees
        Args:
            x: The tensor to rotate
        Returns:
            The rotated tensor
    """

    return tf.image.rot90(x, k=2)


@tf.function
def rotate270(x):
    """
        Rotate the tensor 270 degrees
        Args:
            x: The tensor to rotate
        Returns:
            The rotated tensor
    """

    return tf.image.rot90(x, k=3)


@tf.function
def flip_up_down(x):
    """
        Flips the tensor up down
        Args:
            x: The tensor to flip
        Returns:
            The flipped tensor
    """

    return tf.image.flip_up_down(x)
