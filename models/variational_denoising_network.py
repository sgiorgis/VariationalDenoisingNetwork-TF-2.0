import tensorflow as tf

from tensorflow.keras import Model

from models.denoising_network import DenoisingNetwork
from models.sigma_network import SigmaNetwork


class VariationalDenoisingNetwork(Model):
    def __init__(self, channels=3, filters=64, slope=0.2):
        super(VariationalDenoisingNetwork, self).__init__()
        self.dnet = DenoisingNetwork(channels * 2, filters, slope)
        self.snet = SigmaNetwork(channels * 2, filters, slope)

    @tf.function
    def call(self, inputs):
        return self.dnet(inputs), self.snet(inputs)
