import tensorflow as tf
from math import pi
from tensorflow.keras.metrics import Mean

log_max = tf.math.log(1e4)
log_min = tf.math.log(1e-8)


@tf.function
def psnr_function(denoised_image, clean_image):
    """
        Computes the peak signal to noise ratio (PSNR) for an image.
        Parameters:
            denoised_image (`Tensor): The denoised image clipped by value from 0 to 1
            clean_image (`Tensor): The denoised image clipped by value from 0 to 1
         Returns:
            A `Tensor` containing the mean PSNR value for every batch.
    """
    denoised_image = tf.image.convert_image_dtype(denoised_image, dtype=tf.uint8)
    clean_image = tf.image.convert_image_dtype(clean_image, dtype=tf.uint8)

    psnr = tf.image.psnr(denoised_image, clean_image, max_val=255)

    return tf.reduce_mean(psnr)


@tf.function
def ssim_function(denoised_image, clean_image):
    """
        Computes the mean structural similarity index between clean and denoised image.
        Parameters:
            denoised_image (`Tensor): The denoised image clipped by value from 0 to 1
            clean_image (`Tensor): The clean image clipped by value from 0 to 1
        Returns:
            A `Tensor` containing the mean structural similarity index for every batch.
    """
    denoised_image = tf.image.convert_image_dtype(denoised_image, dtype=tf.uint8)
    clean_image = tf.image.convert_image_dtype(clean_image, dtype=tf.uint8)

    ssim = tf.image.ssim(clean_image, denoised_image, max_val=255)

    return tf.reduce_mean(ssim)


@tf.function
def mse_function(denoised_image, noisy_image):
    """
        Computes the mean squared error between the noisy and denoised image
        Parameters:
            denoised_image (`Tensor): The denoised image clipped by value from 0 to 1
            noisy_image (`Tensor): The noisy image clipped by value from 0 to 1
        Returns:
            A `Tensor` containing the mean squared error for every batch.
    """
    return tf.keras.losses.mse(tf.reshape(denoised_image, [-1]), tf.reshape(noisy_image, [-1]))


@tf.function
def loss_function(dnet_output, snet_output, noisy_image, clean_image, sigma, epsilon, radius=3):
    channels = tf.shape(clean_image)[-1]

    p = 2 * radius + 1
    alpha0 = 0.5 * (p ** 2 - 2)
    beta0 = 0.5 * p ** 2 * sigma

    mean, m2 = extract_dnet_output(channels, dnet_output)
    log_alpha, log_beta = extract_snet_output(channels, snet_output)

    guassian_kl = compute_gaussian_kl_divergence(noisy_image, clean_image, m2, epsilon, mean)
    inverse_gamma_kl = compute_inverse_gamma_kl_divergence(alpha0, beta0, log_alpha, log_beta)
    likelihood = compute_likelihood(log_alpha, log_beta, mean, m2)

    return log_alpha, log_beta, mean, m2, likelihood, guassian_kl, inverse_gamma_kl, likelihood + guassian_kl + inverse_gamma_kl


@tf.function
def extract_dnet_output(channels, dnet_output):
    """
    Extracts the mean and m2 from the denoising network output after it is clipped by value
    Parameters:
        channels: First ':channels' values are the mean and the rest the m2
        dnet_output('Tensor'): The output of denoising network
    """
    dnet_output_clipped = clip_dnet_output(channels, dnet_output)

    return dnet_output_clipped[:, :, :, :channels], tf.math.exp(dnet_output_clipped[:, :, :, channels:])


@tf.function
def extract_snet_output(channels, snet_output):
    """
    Extracts logalpha and logbeta from the sigma network output after it is clipped by value
    Parameters:
        channels: First ':channels' values are the mean and the rest the m2
        snet_output('Tensor'): The output of sigma network
    """
    snet_output_clipped = clip_snet_output(snet_output)
    return snet_output_clipped[:, :, :, :channels], snet_output_clipped[:, :, :, channels:]


@tf.function
def clip_snet_output(snet_output):
    """
        Clips the output of the sigma network.
        Parameters:
            snet_output (`Tensor): The output of the sigma network
        Returns:
            A `Tensor` containing the clipped sigma output from ':log_min' to ':log_max' value
    """
    return tf.clip_by_value(snet_output, log_min, log_max)


@tf.function
def clip_dnet_output(channels, dnet_output):
    """
        Clips the output of the denoising network.
        Parameters:
            channels (`Tensor'): The number of channels of image
            dnet_output (`Tensor): The output of the denoising network
        Returns:
            A `Tensor` containing the clipped denoising output from ':log_min' to ':log_max' value
    """

    out_denoise_slice = dnet_output[:, :, :, :channels]
    out_denoise_clipped_slice = tf.clip_by_value(dnet_output[:, :, :, channels:], log_min, log_max)

    return tf.concat((out_denoise_slice, out_denoise_clipped_slice), axis=channels)


@tf.function
def compute_likelihood(log_alpha, log_beta, mean, m2):
    """
        Computes the likelihood
        Parameters:
            log_alpha: The first output of the sigma network
            log_beta: The second output of the sigma network
            mean: The first output of the denoising network
            m2: The second of the denoising network
        Returns:
            A `Tensor` containing the likelihood
    """

    return 0.5 * tf.math.log(2 * pi) + 0.5 * tf.math.reduce_mean(
        (log_beta - tf.math.digamma(tf.math.exp(log_alpha))) + (mean ** 2 + m2) * tf.math.exp(log_alpha - log_beta))


@tf.function
def compute_inverse_gamma_kl_divergence(alpha0, beta0, log_alpha, log_beta):
    """
        Computes the KL divergence for the inverse Gamma distribution

    """
    alpha = tf.math.exp(log_alpha)
    return tf.math.reduce_mean(
        (alpha - alpha0) * tf.math.digamma(alpha) + (tf.math.lgamma(alpha0) - tf.math.lgamma(alpha)) + alpha0 * (
                log_beta - tf.math.log(beta0)) + beta0 * tf.math.exp(log_alpha - log_beta) - alpha)


@tf.function
def compute_gaussian_kl_divergence(noisy_image, clean_image, m2, epsilon, mean):
    """
        Computes the KL divergence for the variational approximate posterior
    """
    m2_div_eps = tf.math.divide(m2, epsilon ** 2)
    err_mean_gt = noisy_image - clean_image

    return 0.5 * tf.math.reduce_mean(
        (mean - err_mean_gt) ** 2 / epsilon ** 2 + (m2_div_eps - 1 - tf.math.log(m2_div_eps)))
