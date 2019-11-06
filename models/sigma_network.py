import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.initializers import he_normal


class SigmaNetwork(Model):
    def __init__(self, channels, filters=64, slope=0.2):
        super(SigmaNetwork, self).__init__()
        self.block = Sequential()
        self.block.add(Conv2D(filters, kernel_size=3, kernel_initializer=he_normal(seed=10000), padding="SAME"))
        self.block.add(LeakyReLU(alpha=slope))
        self.block.add(Conv2D(filters, kernel_size=3, kernel_initializer=he_normal(seed=10000), padding="SAME"))
        self.block.add(LeakyReLU(alpha=slope))
        self.block.add(Conv2D(filters, kernel_size=3, kernel_initializer=he_normal(seed=10000), padding="SAME"))
        self.block.add(LeakyReLU(alpha=slope))
        self.block.add(Conv2D(filters, kernel_size=3, kernel_initializer=he_normal(seed=10000), padding="SAME"))
        self.block.add(LeakyReLU(alpha=slope))
        self.block.add(Conv2D(channels, kernel_size=3, kernel_initializer=he_normal(seed=10000), padding="SAME"))

    @tf.function
    def call(self, inputs, training=None, mask=None):
        return self.block(inputs)
