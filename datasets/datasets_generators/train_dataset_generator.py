import random
import cv2
import numpy as np
from skimage import img_as_float


class TrainDatasetGenerator:

    def __init__(self, images, noisy_images, dataset_length, patch_size=128, radious=3, epsilon=1e-3):
        self.images = images
        self.noisy_images = noisy_images
        self.dataset_length = dataset_length
        self.patch_size = patch_size
        self.radious = radious
        self.epsilon = epsilon

    def __call__(self):
        """
            Generates all the data points used for training in the benchmark.
            All data points include four tensors (image, noisy_image, sigma_map, epsilon)
            For a random image and it's noisy image it will crop a patch, normalize the images
            and create the sigma map and epsilon.
        Returns:
            The data point for training containing four different tensors
        """
        for i in range(self.dataset_length):
            random_key = random.randint(0, len(self.images) - 1)

            image = self.images[random_key]
            noisy_image = self.noisy_images[random_key]

            clean_image_patch, noisy_image_patch = self.crop(image, noisy_image, self.patch_size)

            clean_image_patch = img_as_float(clean_image_patch)
            noisy_image_patch = img_as_float(noisy_image_patch)

            sigma_map = self.sigma_estimate(noisy_image_patch, clean_image_patch, 2 * self.radious + 1, self.radious)

            yield (
                clean_image_patch,
                noisy_image_patch,
                sigma_map,
                np.ones((1, 1, 1)) * self.epsilon
            )

    def __len__(self):
        return self.dataset_length

    @staticmethod
    def crop(image, noisy_image, patch_size):
        """
            Crops the clean and the noise image with a random path of patch size.
            It will randomly select a patch in the image with the predefined patch size
            Args:
            noisy_image: The noisy image numpy to crop
            image: The clean image numpy to crop
            patch_size: The patch size of the crops to the images
        Returns:
            The cropped clean and noisy images
        """

        height = image.shape[0]
        width = image.shape[1]

        height_start = random.randint(0, height - patch_size)
        width_start = random.randint(0, width - patch_size)

        clean_image_patch = image[height_start:height_start + patch_size, width_start:width_start + patch_size, :]
        noisy_image_patch = noisy_image[height_start:height_start + patch_size, width_start:width_start + patch_size, :]

        return clean_image_patch, noisy_image_patch

    @staticmethod
    def sigma_estimate(noisy_image, clean_image, window, sigma_spatial):
        """
            Taken from: https://github.com/zsyOAOA/VDNet
        """
        noise2 = (noisy_image - clean_image) ** 2

        sigma2_map_est = cv2.GaussianBlur(noise2, (window, window), sigma_spatial)
        sigma2_map_est = sigma2_map_est.astype(np.float32)
        sigma2_map_est = np.where(sigma2_map_est < 1e-10, 1e-10, sigma2_map_est)

        if sigma2_map_est.ndim == 2:
            sigma2_map_est = sigma2_map_est[:, :, np.newaxis]

        return sigma2_map_est
