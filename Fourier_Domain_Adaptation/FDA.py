# Here is the code to apply Fourier Domain Adaptation


# Requirements
import numpy as np
import os

import random
import matplotlib.pyplot as plt
import deeplake


# integrer deeplake


def Fourier_Domain_Adaptation(src_image, target_amplitude, L=0.01, save=False, output_folder="", display=False):

    # Compute FFT of the source image
    fft_src = np.fft.fft2(src_image, axes=(-2, -1))
    amp_src = np.abs(fft_src)
    phase_src = np.angle(fft_src)

    # Shift amplitude spectra to center the low-frequency components
    amp_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    target_amplitude = np.fft.fftshift(target_amplitude, axes=(-2, -1))

    # Defining the square of low amplitudes to be swapped
    _, height, width = amp_src.shape
    radius = int(np.floor(min(height, width) * L))
    center_h = height // 2
    center_w = width // 2

    h_start, h_end = center_h - radius, center_h + radius + 1
    w_start, w_end = center_w - radius, center_w + radius + 1

    amp_src[:, h_start:h_end, w_start:w_end] = target_amplitude[:, h_start:h_end, w_start:w_end]
    amp_src = np.fft.ifftshift(amp_src, axes=(-2, -1))

    # Reconstruct the adapted image
    adapted_fft = amp_src * np.exp(1j * phase_src)
    adapted_image = np.fft.ifft2(adapted_fft, axes=(-2, -1)).real

    # Clip to valid range and convert to uint8 for correct visualisation
    adapted_image = np.clip(adapted_image, 0, 255).astype(np.uint8)

    # saving picture
    if save:
        os.makedirs(output_folder, exist_ok=True)
        filename = f"FDA_L_{L}.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(src_image.transpose((1, 2, 0)).astype(np.uint8))
        axes[0].set_title("Source")
        axes[1].imshow(adapted_image.transpose((1, 2, 0))) # transposition is important for display on pyplot
        axes[1].set_title("FDA → Akoya")
        plt.suptitle(f"Global FDA L={L}")
        plt.savefig(os.path.join(output_folder, filename))
        plt.close(fig)
    
    if display:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(src_image.transpose((1, 2, 0)).astype(np.uint8))
        axes[0].set_title("Source")
        axes[1].imshow(adapted_image.transpose((1, 2, 0))) # transposition is important for display on pyplot
        axes[1].set_title("FDA → Akoya")
        plt.suptitle(f"Global FDA L={L}")
        plt.show()


    return adapted_image


# Example on how to run the function




    