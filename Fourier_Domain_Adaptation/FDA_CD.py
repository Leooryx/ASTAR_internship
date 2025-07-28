# TODO: 1) gerer la cdf du input scanner, 2) optimiser le code (simplement)



# Requirements


import numpy as np
import deeplake
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import random
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import joblib
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

LABEL_MAP = {
    "Stroma": 0,
    "Normal": 1,
    "G3":     2,
    "G4":     3,
    "G5":     4
}

REVERSE_LABEL_MAP = {0: "Stroma",
    1: "Normal", 
    2: "G3",
    3: "G4",
    4: "G5"
}

classifier = joblib.load("/home/leolr-int/nfs/transformed_data/weights/random_forest.pkl")
distrib_Akoya = joblib.load("/home/leolr-int/nfs/transformed_data/weights/cdf_Akoya.pkl")
distrib_KFBio = joblib.load("/home/leolr-int/nfs/transformed_data/weights/cdf_KFBio.pkl")

# the distribution of KFBio should be calculated during inference: in real-life usage, 
# the WSI will be given and the distribution of low amplitudes from patches should be calculated over this image only 


def Flat_log_Fourier(src_image, L=0.01):
    
    # Compute FFT of the source image
    fft_src = np.fft.fft2(src_image, axes=(-2, -1))
    amp_src = np.abs(fft_src)
    phase_src = np.angle(fft_src)

    # Shift amplitude spectra to center the low-frequency components
    amp_src = np.fft.fftshift(amp_src, axes=(-2, -1))

    # Defining the square of low amplitudes to be swapped
    _, height, width = amp_src.shape
    radius = int(np.floor(min(height, width) * L))
    center_h = height // 2
    center_w = width // 2

    h_start, h_end = center_h - radius, center_h + radius + 1
    w_start, w_end = center_w - radius, center_w + radius + 1

    small_amp = amp_src[:, h_start:h_end, w_start:w_end]

    return np.log(small_amp.reshape(-1))



def match_vector_to_target_distribution(vector, source_cdfs, target_cdfs):
    """
    Transforms a 75-dimensional vector from source distribution to match the target distribution.
    
    Parameters:
    - vector: np.ndarray of shape (75,)
    - source_cdfs: dict[label][feature_index] = (xs, cdf_vals)
    - target_cdfs: same structure as source_cdfs
    - Returns: transformed vector (np.ndarray of shape (75,))
    """
    #assert vector.shape[0] == 75, "Input vector must have 75 dimensions"

    transformed_vector = np.zeros_like(vector)

    for i in range(75):
        xs_src, cdf_src = source_cdfs[i]
        xs_tgt, cdf_tgt = target_cdfs[i]

        # Interpolator for source: value -> probability
        cdf_interp_src = interp1d(xs_src, cdf_src, bounds_error=False, fill_value=(0.0, 1.0))

        # Interpolator for target: probability -> value
        inv_cdf_interp_tgt = interp1d(cdf_tgt, xs_tgt, bounds_error=False, fill_value="extrapolate")

        # Step 1: get cumulative probability of vector[i] under source
        p = float(cdf_interp_src(vector[i]))

        # Step 2: map to value under target
        x_new = float(inv_cdf_interp_tgt(p))

        transformed_vector[i] = x_new

    return transformed_vector



def get_white_pixel_mask(image, white_thresh=200):
    """
    Returns a binary mask of white pixels.
    Input: image (C, H, W), RGB
    Output: mask (H, W), dtype=bool
    """
    img = np.transpose(image, (1, 2, 0))  # to HWC
    #designates the pixel as white if and only if all three colors are above 200
    return np.all(img > white_thresh, axis=-1)

def new_FDA(src_image, amp_matched, L=0.01, save=False, output_folder="", display=False):
    
    # Store mask to apply it to the output
    white_mask = get_white_pixel_mask(src_image, white_thresh=200)  # (H, W) coordinates of white pixel
    white_pixel_values = src_image[:, white_mask]  # shape: (C, N_white_pixels)

    # reference point, average amplitude from Akoya
    target_amplitude = np.load("/home/leolr-int/nfs/ASTAR_internship/Fourier_Domain_Adaptation/stored_amplitude/general_average_akoya.npy")

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

    ## Change here with the new vector of amplitudes!
    
    #exponential is important because i have used logarithms 
    amp_src[:, h_start:h_end, w_start:w_end] = np.exp(amp_matched.reshape(3,5,5))
    
    amp_src = np.fft.ifftshift(amp_src, axes=(-2, -1))

    # Reconstruct the adapted image
    adapted_fft = amp_src * np.exp(1j * phase_src)
    adapted_image = np.fft.ifft2(adapted_fft, axes=(-2, -1)).real

    # Clip to valid range and convert to uint8 for correct visualisation
    adapted_image = np.clip(adapted_image, 0, 255).astype(np.uint8)

    # Put white pixels from original image back in FDA output
    for c in range(src_image.shape[0]):
        channel = adapted_image[c]
        channel[white_mask] = white_pixel_values[c]
        adapted_image[c] = channel

    # saving picture
    if save:
        os.makedirs(output_folder, exist_ok=True)
        filename = f"WFDA_L_{L}.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(src_image.transpose((1, 2, 0)).astype(np.uint8))
        axes[0].set_title("Source")
        axes[1].imshow(adapted_image.transpose((1, 2, 0)))
        axes[1].set_title("FWDA → Akoya")
        plt.suptitle(f"Fourier Domain Adaptation with white filter and amplitude distribution matching to Akoya L={L}")
        plt.savefig(os.path.join(output_folder, filename))
        plt.close(fig)
    
    if display:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(src_image.transpose((1, 2, 0)).astype(np.uint8))
        axes[0].set_title("Source")
        axes[1].imshow(adapted_image.transpose((1, 2, 0)))
        axes[1].set_title("WFDA → Akoya")
        plt.suptitle(f"Fourier Domain Adaptation with white filter and amplitude distribution matching to Akoya L={L}")
        plt.show()

    return adapted_image



def FDA_CD(src_img, save=False, output_folder="", display=False):
    
    vector = Flat_log_Fourier(src_img)
    
    pred_label = int(classifier.predict(vector.reshape(1, -1))) 
    # because transposition happened during creation of amplitudes dataset

    src_cdfs = distrib_KFBio[pred_label]
    tgt_cdfs = distrib_Akoya[pred_label]

    vector_KFBio_to_Akoya = match_vector_to_target_distribution(vector, src_cdfs, tgt_cdfs)

    return new_FDA(src_img, vector_KFBio_to_Akoya,0.01, save, output_folder, display)


# Test

import time


dataset_path_KFbio_1 = f"/home/leolr-int/nfs/data/data/patched/dim_256/Train/Subset3_Train_1_KFBio"
KFBio_1 = deeplake.open_read_only(dataset_path_KFbio_1)

# Preprocessing
src_img = KFBio_1[200]["patch"].transpose((2, 0, 1))  # (3, 256, 256)
#trg_img = akoya_1[200]["patch"].transpose((2, 0, 1))

start_time = time.time()
KFBio_to_Akoya = FDA_CD(src_img, save=False, output_folder="", display=False)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds") 
#0.04311990737915039 seconds

