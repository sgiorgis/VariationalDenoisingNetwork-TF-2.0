import csv
import os


def save_losses(epoch, losses, checkpoint_directory):
    with open(os.path.join(checkpoint_directory, 'loss_history.csv'), mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        losses = [[
            epoch,
            float(losses['train_loss'].result()),
            float(losses['train_mse'].result()),
            float(losses['validation_mse'].result()),
            float(losses['validation_psnr'].result()),
            float(losses['validation_ssim'].result())
        ]]
        writer.writerows(losses)


def save_simulation_losses(epoch, losses, checkpoint_directory):
    with open(os.path.join(checkpoint_directory, 'loss_history.csv'), mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        losses = [[
            epoch,
            float(losses['train_loss'].result()),
            float(losses['train_mse'].result()),
            float(losses['train_psnr'].result()),
            float(losses['train_ssim'].result())
        ]]
        writer.writerows(losses)
