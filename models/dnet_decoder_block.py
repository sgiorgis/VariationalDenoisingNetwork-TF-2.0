import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2DTranspose, Concatenate
from tensorflow.keras.initializers import he_normal

from models.dnet_encoder_block import DNetEncoderBlock


class DNetDecoderBlock(Model):
    def __init__(self, filters, kernel_size, slope=0.2):
        super(DNetDecoderBlock, self).__init__()
        self.decoder_block = Conv2DTranspose(
            filters,
            kernel_size=kernel_size,
            kernel_initializer=he_normal(seed=10000),
            strides=2,
            padding="same"
        )
        self.encoder_block = DNetEncoderBlock(filters, kernel_size=3, slope=slope)

    @tf.function
    def call(self, x, encoder_block):
        x = self.decoder_block(x)
        concat = Concatenate()
        crop1 = self.center_crop(encoder_block, x.shape[1:3])
        out = concat([x, crop1])
        out = self.encoder_block(out)
        return out

    @staticmethod
    def center_crop(layer, target_size):
        _, layer_height, layer_width, _ = layer.shape
        diff_height = (layer_height - target_size[0]) // 2
        diff_width = (layer_width - target_size[1]) // 2
        return layer[:, diff_height:(diff_height + target_size[0]), diff_width:(diff_width + target_size[1]), :]
