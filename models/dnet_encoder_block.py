import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import LeakyReLU, Conv2D
from tensorflow.keras.initializers import he_normal


class DNetEncoderBlock(Model):

    def __init__(self, filters, kernel_size, slope=0.2):
        super(DNetEncoderBlock, self).__init__()

        self.block = Sequential()
        self.block.add(Conv2D(filters, kernel_size=kernel_size, kernel_initializer=he_normal(seed=10000), padding="SAME"))
        self.block.add(LeakyReLU(alpha=slope))
        self.block.add(Conv2D(filters, kernel_size=kernel_size, kernel_initializer=he_normal(seed=10000), padding="SAME"))
        self.block.add(LeakyReLU(alpha=slope))

    @tf.function
    def call(self, inputs, training=None, mask=None):
        return self.block(inputs)
