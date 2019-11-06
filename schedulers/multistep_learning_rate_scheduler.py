import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import LearningRateSchedule


class MultiStepLearningRateScheduler(LearningRateSchedule):
    def __init__(self, initial_learning_rate, gamma=0.5):
        super(MultiStepLearningRateScheduler, self).__init__()
        self.learning_rate = tf.cast(initial_learning_rate, tf.float32)
        self.gamma = tf.cast(gamma, tf.float32)

    @tf.function
    def __call__(self, step):
        """
            The scheduler will linearly decay the learning rate every 10 epochs
            Params:
                step: The current step representing the epoch in our implementation
        """
        if tf.math.mod(step, math_ops.cast(10, tf.float32)) == 0:
            self.learning_rate = tf.math.multiply(self.learning_rate, self.gamma)

        return self.learning_rate

    def get_config(self):
        pass
