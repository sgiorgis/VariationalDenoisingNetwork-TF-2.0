import random

import numpy as np
from skimage import img_as_float


class SimulateTrainGenerator:

    def __init__(self, images, dataset_length, patch_size=128, radious=3, epsilon=0.007071067811865475):
        self.images = images
        self.dataset_length = dataset_length
        self.patch_size = patch_size
        self.radious = radious
        self.epsilon = epsilon

    def __call__(self):
        """
            Generates all the data points used for training in the simulation.
            All data points include four tensors (image, noisy_image, sigma_map, epsilon)
            For a random image it will generate the noise, it will crop a patch, normalize the images
            and create the sigma map and epsilon.
            Returns:
                The data point for training containing four different tensors
        """
        for key in range(self.dataset_length):
            random_key = random.randint(0, len(self.images) - 1)

            image = self.images[random_key]
            image = self.crop(image, self.patch_size)
            image = img_as_float(image)

            sigma = SimulateTrainGenerator.generate_sigma()

            noise = np.random.normal(size=image.shape) * sigma
            noisy_image = image + noise.astype(np.float32)

            sigma_map = np.square(sigma)
            sigma_map = np.tile(sigma_map, (1, 1, 3))
            sigma_map = np.where(sigma_map < 1e-10, 1e-10, sigma_map)

            yield (
                image,
                noisy_image,
                sigma_map,
                np.ones((1, 1, 1)) * self.epsilon
            )

    def __len__(self):
        return self.dataset_length

    @staticmethod
    def crop(image, patch_size):
        """
            Crops the clean image with a random path of patch size.
            It will randomly select a patch in the image with the predefined patch size
            Args:
            image: The clean image numpy to crop
            patch_size: The patch size of the crops to the images
        Returns:
            The cropped clean image
        """
        height, width, _ = image.shape

        height_start = random.randint(0, height - patch_size)
        width_start = random.randint(0, width - patch_size)

        clean_image_patch = image[height_start:height_start + patch_size, width_start:width_start + patch_size, :3]

        return clean_image_patch

    @staticmethod
    def generate_sigma():
        """
           Taken from: https://github.com/zsyOAOA/VDNet
        """
        temp_means = [random.uniform(0, 128), random.uniform(0, 128)]
        temp_sigma = random.uniform(0 + 32, 128 - 32)

        kernel = SimulateTrainGenerator.gaussian_kernel(128, 128, temp_means, temp_sigma)

        up = random.uniform(0 / 255.0, 75 / 255.0)
        down = random.uniform(0 / 255.0, 75 / 255.0)

        if up < down:
            up, down = down, up

        up += 5 / 255.0

        sigma_map = down + (kernel - kernel.min()) / (kernel.max() - kernel.min()) * (up - down)
        sigma_map = sigma_map.astype(np.float32)

        return sigma_map[:, :, np.newaxis]

    @staticmethod
    def gaussian_kernel(height, width, temp_means, temp_sigma):
        """
            Taken from https://github.com/zsyOAOA/VDNet
        """
        center_height = temp_means[0]
        center_width = temp_means[1]

        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        ZZ = 1. / (2 * np.pi * temp_sigma ** 2) * np.exp(
            (-(xx - center_height) ** 2 - (yy - center_width) ** 2) / (2 * temp_sigma ** 2))

        return ZZ
