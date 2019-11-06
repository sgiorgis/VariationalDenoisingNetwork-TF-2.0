import h5py
import numpy as np
from skimage import img_as_float


class ValidationDatasetGenerator:
    def __init__(self, images, noisy_images):
        self.images = h5py.File(images, 'r')
        self.noisy_images = h5py.File(noisy_images, 'r')
        self.keys = self.images.keys()

    def __call__(self):
        """
            Generates all the data points used for validation in the benchmark.
            All data points include two tensors (image, noisy_image)
            For every different key in the hdf5 file it will normalize the images
            to floating point format and return the data point.
            Returns:
                The data point for validation containing two different tensors
        """
        for key in self.keys:
            clean_image = np.array(self.images[key])
            noisy_image = np.array(self.noisy_images[key])

            clean_image = img_as_float(clean_image)
            noisy_image = img_as_float(noisy_image)

            yield (
                clean_image,
                noisy_image,
            )

    def __len__(self):
        return len(self.keys)
