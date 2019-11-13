import os
import sys

sys.path.append(os.getcwd())

import tensorflow as tf

from glob import glob

from tensorflow_core.python.keras.metrics import Mean
from tqdm import tqdm

from arguments import parse_train_arguments
from datasets.dataset_loader import load_train_data, load_validation_dataset
from losses.loss import loss_function, mse_function, ssim_function, psnr_function
from models.model_loader import load_model, save_model
from summaries import save_losses
from utils import manage_gpu_memory_usage

@tf.function
def clip_gradients(gradients, clip_norms):
    clipped_gradients_dnet, dnet_new_norm = tf.clip_by_global_norm(gradients[:36], clip_norms[0])
    clipped_gradients_snet, snet_new_norm = tf.clip_by_global_norm(gradients[36:], clip_norms[1])

    return clipped_gradients_dnet, dnet_new_norm, clipped_gradients_snet, snet_new_norm


@tf.function
def train_step(model, optimizer, data, losses, clip_norms, radious):
    clean_image = data[0]
    noisy_image = data[1]
    sigma = data[2]
    epsilon = data[3]

    with tf.GradientTape() as tape:
        dnet_output, snet_output = model(noisy_image)

        log_alpha, log_beta, mean, m2, likelihood, guassian_kl, inverse_gamma_kl, loss = loss_function(
            dnet_output=dnet_output,
            snet_output=snet_output,
            noisy_image=noisy_image,
            clean_image=clean_image,
            sigma=sigma,
            epsilon=epsilon,
            radius=radious
        )

        gradients = tape.gradient(loss, model.trainable_variables)

        clipped_gradients_dnet, dnet_new_norm, clipped_gradients_snet, snet_new_norm = clip_gradients(gradients,
                                                                                                      clip_norms)
        optimizer.apply_gradients(zip(clipped_gradients_dnet + clipped_gradients_snet, model.trainable_variables))

        denoised_image = tf.clip_by_value(noisy_image - dnet_output[:, :, :, :3], 0, 1)

        mse = mse_function(denoised_image, clean_image)

        losses['train_loss'](loss)
        losses['train_mse'](mse)

        return dnet_new_norm, snet_new_norm


@tf.function
def validation_step(model, data, losses):
    clean_image = data[0]
    noisy_image = data[1]

    dnet_output, snet_output = model(noisy_image)

    # clip the value so the values match the input (0,1)
    denoised_image = tf.clip_by_value(noisy_image - dnet_output[:, :, :, :3], 0, 1)

    losses['validation_mse'](mse_function(denoised_image, clean_image))
    losses['validation_ssim'](ssim_function(denoised_image, clean_image))
    losses['validation_psnr'](psnr_function(denoised_image, clean_image))


def on_batch_end(epoch, index, dnet_new_norm, snet_new_norm, total_clip_norms, losses, progress_bar, batches):
    progress_bar.set_description(
        'Epoch {} iteration {}/{} | Loss {:.3f}'.format(epoch, index, batches, losses['train_loss'].result()))

    total_clip_norms[0] = (total_clip_norms[0] * (index / (index + 1)) + dnet_new_norm / (index + 1))
    total_clip_norms[1] = (total_clip_norms[1] * (index / (index + 1)) + snet_new_norm / (index + 1))


def on_epoch_end(model, optimizer, epoch, losses, best_losses, clip_norms, total_clip_norms, checkpoint_directory):
    save_losses(epoch, losses, checkpoint_directory)

    # save the model for every best validation_psnr result
    if best_losses['validation_psnr'] < losses['validation_psnr'].result():
        best_losses['validation_psnr'] = losses['validation_psnr'].result()
        best_losses['validation_ssim'] = losses['validation_ssim'].result()
        save_model(checkpoint_directory, model, optimizer, epoch, clip_norms)

    for loss in losses.values():
        loss.reset_states()

    clip_norms[0] = min(clip_norms[0], total_clip_norms[0])
    clip_norms[1] = min(clip_norms[1], total_clip_norms[1])


def train():
    device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'

    args = parse_train_arguments()

    with tf.device(device):

        best_losses = {
            'validation_psnr': 0,
            'validation_ssim': 0
        }

        losses = {
            'train_loss': Mean(name='train_loss'),
            'train_mse': Mean(name='train_mse'),
            'validation_mse': Mean(name='validation_mse'),
            'validation_psnr': Mean(name='validation_psnr'),
            'validation_ssim': Mean(name='validation_ssim')
        }

        train_dataset_path = glob(os.path.join(args.train_dataset_base_path, '**/*GT*.PNG'), recursive=True)
        train_noisy_dataset_path = [path_image.replace('GT', 'NOISY') for path_image in train_dataset_path]

        dataset = load_train_data(
            train_dataset_path=train_dataset_path,
            train_noisy_dataset_path=train_noisy_dataset_path,
            data_length=args.batch_size * args.batches,
            patch_size=args.patch_size,
            radious=args.radious,
            epsilon=args.epsilon
        )

        validation_dataset = load_validation_dataset(
            validation_clean_dataset_path=args.validation_clean_dataset_path,
            validation_noisy_dataset_path=args.validation_noisy_dataset_path
        )

        model, optimizer, initial_epoch, clip_norms = load_model(
            checkpoint_directory=args.checkpoint_directory,
            restore_model=args.restore_model,
            learning_rate=args.learning_rate
        )

        for epoch in range(initial_epoch, args.epochs + 1):

            total_clip_norms = [tf.cast(0, dtype=tf.float32), tf.cast(0, dtype=tf.float32)]
            batched_dataset = dataset.batch(args.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            progress_bar = tqdm(batched_dataset, total=args.batches)

            for index, data_batch in enumerate(progress_bar):
                dnet_new_norm, snet_new_norm = train_step(model, optimizer, data_batch, losses, clip_norms,
                                                          args.radious)
                on_batch_end(epoch, index, dnet_new_norm, snet_new_norm, total_clip_norms, losses, progress_bar,
                             args.batches)

            validation_progress_bar = tqdm(validation_dataset.batch(args.batch_size))

            for validation_data_batch in validation_progress_bar:
                validation_step(model, validation_data_batch, losses)

            on_epoch_end(model, optimizer, epoch, losses, best_losses, clip_norms, total_clip_norms, args.checkpoint_directory)


if __name__ == '__main__':
    manage_gpu_memory_usage()
    train()
