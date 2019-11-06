import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.initializers import he_normal

from models.dnet_decoder_block import DNetDecoderBlock
from models.dnet_encoder_block import DNetEncoderBlock


class DenoisingNetwork(Model):
    def __init__(self, channels=6, filters=32, slope=0.2):
        super(DenoisingNetwork, self).__init__()

        self.encoder_stack = [
            DNetEncoderBlock(filters, 3, slope),
            DNetEncoderBlock(filters * 2, 3, slope),
            DNetEncoderBlock(filters * 4, 3, slope),
            DNetEncoderBlock(filters * 8, 3, slope)
        ]

        self.decoder_stack = [
            DNetDecoderBlock(filters * 4, 2, slope),
            DNetDecoderBlock(filters * 2, 3, slope),
            DNetDecoderBlock(filters, 3, slope)
        ]

        self.last = Conv2D(channels, kernel_initializer=he_normal(seed=10000), kernel_size=3, use_bias=True, padding="SAME")

    @tf.function
    def call(self, inputs, training=None, mask=None):
        encoder_blocks = []
        for i, encoder_block in enumerate(self.encoder_stack):
            inputs = encoder_block(inputs)
            # last bottleneck block will not be added and will not be avg pooled
            if i != len(self.encoder_stack) - 1:
                encoder_blocks.append(inputs)
                inputs = tf.nn.avg_pool2d(inputs, ksize=2, strides=2, padding="SAME")

        for i, decoder_block in enumerate(self.decoder_stack):
            inputs = decoder_block(inputs, encoder_blocks[-i - 1])

        return self.last(inputs)
