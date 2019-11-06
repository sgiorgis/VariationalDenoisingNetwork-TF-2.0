import tensorflow as tf
from tensorflow_core.python.keras.metrics import Mean

from tqdm import tqdm
from arguments import parse_test_arguments
from datasets.dataset_loader import load_validation_dataset
from losses.loss import mse_function, ssim_function, psnr_function
from models.model_loader import load_test_model


def validate(model, data, losses):
    clean_image = data[0]
    noisy_image = data[1]

    dnet_output, snet_output = model(noisy_image)

    # clip the value so the values match the input (0,1)
    denoised_image = tf.clip_by_value(noisy_image - dnet_output[:, :, :, :3], 0, 1)

    losses['validation_mse'](mse_function(denoised_image, clean_image))
    losses['validation_ssim'](ssim_function(denoised_image, clean_image))
    losses['validation_psnr'](psnr_function(denoised_image, clean_image))


def test():
    args = parse_test_arguments()

    model = load_test_model(args.checkpoint_directory)

    validation_dataset = load_validation_dataset(
        validation_clean_dataset_path=args.validation_clean_dataset_path,
        validation_noisy_dataset_path=args.validation_noisy_dataset_path
    )

    losses = {
        'validation_mse': Mean(name='validation_mse'),
        'validation_psnr': Mean(name='validation_psnr'),
        'validation_ssim': Mean(name='validation_ssim')
    }

    validation_progress_bar = tqdm(validation_dataset.batch(args.batch_size))

    for validation_data_batch in validation_progress_bar:
        validate(model, validation_data_batch, losses)

    for loss in losses.keys():
        print("{}:{}".format(loss, float(losses[loss].result())))


if __name__ == '__main__':
    test()
