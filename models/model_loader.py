import os

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from models.variational_denoising_network import VariationalDenoisingNetwork
from schedulers.multistep_learning_rate_scheduler import MultiStepLearningRateScheduler


def load_test_model(checkpoint_directory):
    """
        Loads the test model from the checkpoint directory.
        Clips norms, optimizer and initial epoch are not needed since it's only
        used for testing. It will by default load the last checkpoint unless it is
        configured otherwise in the checkpoint file.
        Params:
            checkpoint_directory: The directory containing the checkpoints.
    """
    clip_norms = tf.Variable([tf.cast(1e4, dtype=tf.float32), tf.cast(1e3, dtype=tf.float32)])
    model = VariationalDenoisingNetwork()
    optimizer = Adam(learning_rate=MultiStepLearningRateScheduler(initial_learning_rate=4e-4))
    initial_epoch = tf.Variable(1)

    checkpoint = tf.train.Checkpoint(step=initial_epoch, optimizer=optimizer, net=model, clip_norms=clip_norms)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=3)

    if manager.latest_checkpoint is None:
        raise Exception("Checkpoint directory does not contain a valid checkpoint")

    checkpoint.restore(manager.latest_checkpoint)

    return model


def load_model(checkpoint_directory, restore_model, learning_rate):
    """
        Loads the model for training from the checkpoint directory
        It will by default load the last checkpoint unless it is
        configured otherwise in the checkpoint file. If the :restore_model
        is True it will restore the model, otherwise it will return a newly initialized one.
        Params:
            checkpoint_directory: The directory containing the checkpoints.
            restore_model: Will restore model is True
            learning_rate: The learning rate to initialize the model with
    """
    clip_norms = tf.Variable([tf.cast(1e4, dtype=tf.float32), tf.cast(1e3, dtype=tf.float32)])
    model = VariationalDenoisingNetwork()
    optimizer = Adam(learning_rate=MultiStepLearningRateScheduler(learning_rate))
    initial_epoch = tf.Variable(0)

    if restore_model:
        checkpoint = tf.train.Checkpoint(step=initial_epoch, optimizer=optimizer, net=model, clip_norms=clip_norms)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=3)

        if manager.latest_checkpoint is None:
            raise Exception("Checkpoint directory does not contain a valid checkpoint")

        checkpoint.restore(manager.latest_checkpoint)

    return model, optimizer, int(initial_epoch.value()) + 1, list(clip_norms.value())


def save_model(checkpoint_directory, model, optimizer, epoch, clip_norms):
    """
        Saves a checkpoint of the model for the current epoch
        Params:
            checkpoint_directory: The directory containing the checkpoints.
            model: The model object to save
            optimizer: The optimizer to save
            epoch: The epoch to save as step
            clip_norms: The clip_norms to save
    """
    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(epoch),
        optimizer=optimizer,
        net=model,
        clip_norms=tf.Variable(clip_norms)

    )
    checkpoint.save(os.path.join(checkpoint_directory, "{}_epoch".format(epoch)))
