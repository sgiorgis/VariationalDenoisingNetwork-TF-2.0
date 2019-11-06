import os
from glob import glob
import tensorflow as tf
from tensorflow_core.python.keras.metrics import Mean
from tqdm import tqdm

from arguments import parse_train_arguments
from datasets.dataset_loader import load_simulation_data
from losses.loss import loss_function, mse_function, psnr_function, ssim_function
from models.model_loader import load_model, save_model
from summaries import save_simulation_losses
from utils import manage_gpu_memory_usage


@tf.function
def clip_gradients(gradients, clip_norms):
    clipped_gradients_dnet, dnet_new_norm = tf.clip_by_global_norm(gradients[:36], clip_norms[0])
    clipped_gradients_snet, snet_new_norm = tf.clip_by_global_norm(gradients[36:], clip_norms[1])

    return clipped_gradients_dnet, dnet_new_norm, clipped_gradients_snet, snet_new_norm


@tf.function
def train_step(model, optimizer, data, losses, clip_norms, radious):
    noisy_image = data[1]
    clean_image = data[0]
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

        clipped_gradients_dnet, dnet_new_norm, clipped_gradients_snet, snet_new_norm = clip_gradients(
            gradients,
            clip_norms
        )

        optimizer.apply_gradients(zip(clipped_gradients_dnet + clipped_gradients_snet, model.trainable_variables))

        denoised_image = tf.clip_by_value(noisy_image - dnet_output[:, :, :, :3], 0, 1)
        mse = mse_function(denoised_image, clean_image)
        psnr = psnr_function(denoised_image, clean_image)
        ssim = ssim_function(denoised_image, clean_image)

        losses['train_loss'](loss)
        losses['train_mse'](mse)
        losses['train_psnr'](psnr)
        losses['train_ssim'](ssim)

        return dnet_new_norm, snet_new_norm


def on_batch_end(epoch, index, dnet_new_norm, snet_new_norm, total_clip_norms, losses, progress_bar):
    progress_bar.set_description(
        'Epoch {} iteration {}/{} | Loss {:.3f}'.format(epoch, index, 5000, losses['train_loss'].result()))

    total_clip_norms[0] = (total_clip_norms[0] * (index / (index + 1)) + dnet_new_norm / (index + 1))
    total_clip_norms[1] = (total_clip_norms[1] * (index / (index + 1)) + snet_new_norm / (index + 1))


def on_epoch_end(model, optimizer, epoch, losses, clip_norms, total_clip_norms, checkpoint_directory, checkpoint_frequency):
    save_simulation_losses(epoch, losses, checkpoint_directory)

    # we have not validation data here so we defined the best checkpoint_frequency based on experiments
    if epoch % checkpoint_frequency == 0:
        save_model(checkpoint_directory, model, optimizer, epoch, clip_norms)

    for loss in losses.values():
        loss.reset_states()

    clip_norms[0] = min(clip_norms[0], total_clip_norms[0])
    clip_norms[1] = min(clip_norms[1], total_clip_norms[1])


def train():
    device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'
    args = parse_train_arguments()

    with tf.device(device):

        losses = {
            'train_loss': Mean(name='train_loss'),
            'train_mse': Mean(name='train_mse'),
            'train_psnr': Mean(name='train_psnr'),
            'train_ssim': Mean(name='train_ssim')
        }

        train_dataset_path = glob(os.path.join(args.train_dataset_base_path, '**/**.png'), recursive=True) + \
                             glob(os.path.join(args.train_dataset_base_path, '**/**.jpg'), recursive=True) + \
                             glob(os.path.join(args.train_dataset_base_path, '**/**.bmp'), recursive=True)

        dataset = load_simulation_data(train_dataset_path, args.batch_size * args.batches, args.patch_size,
                                       args.radious, args.epsilon)

        model, optimizer, initial_epoch, clip_norms = load_model(args.checkpoint_directory, args.restore_model,
                                                                 args.learning_rate)

        for epoch in range(initial_epoch, args.epochs):
            total_clip_norms = [tf.cast(0, dtype=tf.float32), tf.cast(0, dtype=tf.float32)]
            batched_dataset = dataset.batch(args.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            progress_bar = tqdm(batched_dataset, total=args.batches)

            for index, data_batch in enumerate(progress_bar):
                dnet_new_norm, snet_new_norm = train_step(model, optimizer, data_batch, losses, clip_norms,
                                                          args.radious)
                on_batch_end(epoch, index, dnet_new_norm, snet_new_norm, total_clip_norms, losses, progress_bar)

            on_epoch_end(model, optimizer, epoch, losses, clip_norms, total_clip_norms, args.checkpoint_directory, args.checkpoint_frequency)


if __name__ == '__main__':
    manage_gpu_memory_usage()
    train()
