## **Description of the file**
# This file contains all the functions needed to perform the Fourier Domain Adaptation - Conditional & Distributional
# It contains the functions to extract the low amplitudes vectors
# to retrieve the low amplitudes distributions
# the distribution match
# the low amplitude swap
# the white filter to avoid artifacts

## Inefficiency found: the vector of amplitude for the input patch is computed two times during the computations of distributions and the final transformation


## Requirements

import numpy as np
import deeplake
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d
import random
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

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




## Extract the vectors of the log of the 1% lowest amplitudes
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


## Generate a dataset of the amplitude vectors to produce the target distribution or to train classifiers
# specify 'group' (WSI) to avoid data leakage!
def _process_fourier(img_L):
    """Helper pour le multiprocessing."""
    img, L = img_L
    return Flat_log_Fourier(img, L)

def extract_features_parallel(imgs, L, max_workers=4):
    """Extrait les features en parallèle pour une liste d’images."""
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_process_fourier, [(img, L) for img in imgs]))

def load_data(input_root, train_or_test, L=0.01, max_workers=4, binary=False):
    X, y, groups = [], [], []

    for scanner in list_train_scanners:
        for i in range(1, 27):  
            dataset_name = f"Subset3_{train_or_test}_{i}_{scanner}"
            dataset_path = f"{input_root}/{train_or_test}/{dataset_name}"
            if not os.path.exists(dataset_path):
                continue

            print(f"Loading {dataset_path}")
            dataset = deeplake.open_read_only(dataset_path)

            imgs, labels, group_list = [], [], []

            for sample in tqdm(dataset, desc=dataset_name):
                patch = sample["patch"]
                if patch.shape != (256, 256, 3):
                    continue

                img = patch.transpose(2, 0, 1)
                label = int(sample["label"]) #modify here for binary classification

                imgs.append(img)
                labels.append(label)
                group_list.append(dataset_name)

            if imgs:
                features = extract_features_parallel(imgs, L, max_workers=4)
                X.extend(features)
                y.extend(labels)
                groups.extend(group_list)

    return np.array(X), np.array(y), np.array(groups)


## Compute the distribution of the low amplitudes
# non binary distributions per coordinate
# for each label, we compute the dsitribution of each coordinate inside the amplitude vector. 
def compute_smoothed_cdfs(X_train, y_train, num_points=512):
    labels = np.unique(y_train)
    n_features = X_train.shape[1]
    cdf_dict = {}

    for label in tqdm(labels, desc="Processing labels"):
        X_label = X_train[y_train == label]
        cdf_dict[label] = {}

        for feature_idx in range(n_features):
            values = X_label[:, feature_idx]

            # Fit KDE
            kde = gaussian_kde(values)

            # Evaluate KDE on linspace covering the data range
            xmin, xmax = np.percentile(values, [0.5, 99.5])  # robust range
            xs = np.linspace(xmin, xmax, num_points)
            pdf = kde(xs)

            # Normalize to CDF
            cdf = np.cumsum(pdf)
            cdf /= cdf[-1]  # Normalize to [0,1]

            cdf_dict[label][feature_idx] = (xs, cdf)

    return cdf_dict



## Distribution match to generate a new amplitude vector
def match_vector_to_target_distribution(vector, source_cdfs, target_cdfs):
    """
    Transforms a 75-dimensional vector from source distribution to match the target distribution.
    
    Parameters:
    - vector: np.ndarray of shape (75,)
    - source_cdfs: dict[label][coordinate] = (xs, cdf_vals)
    - target_cdfs: same structure as source_cdfs
    - Returns: transformed vector (np.ndarray of shape (75,))
    """

    transformed_vector = np.zeros_like(vector)

    for i in range(75): #loop over each coordinate
        xs_src, cdf_src = source_cdfs[i]
        xs_tgt, cdf_tgt = target_cdfs[i]

        # Interpolator for source: value -> probability
        cdf_interp_src = interp1d(xs_src, cdf_src, bounds_error=False, fill_value=(0.0, 1.0))

        # Interpolator for target: probability -> value
        inv_cdf_interp_tgt = interp1d(cdf_tgt, xs_tgt, bounds_error=False, fill_value="extrapolate")

        # Distribution match
        # Get cumulative probability of vector[i] under source
        p = float(cdf_interp_src(vector[i]))
        # Map to value under target
        x_new = float(inv_cdf_interp_tgt(p))

        transformed_vector[i] = x_new

    return transformed_vector


## White filter
def get_white_pixel_mask(image, white_thresh=200):
    """
    Returns a binary mask of white pixels.
    Input: image (C, H, W), RGB
    Output: mask (H, W), dtype=bool
    """
    img = np.transpose(image, (1, 2, 0))  # to HWC
    #designates the pixel as white if and only if all three colors are above 200
    return np.all(img > white_thresh, axis=-1)


## Amplitude swap
def amplitude_swap(src_image, amp_matched, L=0.01, save=False, output_folder="", display=False):
    
    # Store mask to apply it to the output
    white_mask = get_white_pixel_mask(src_image, white_thresh=200)  # (H, W) coordinates of white pixel
    white_pixel_values = src_image[:, white_mask]  # shape: (C, N_white_pixels)
    
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
        filename = f"FDA_CD_L={L}.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(src_image.transpose((1, 2, 0)).astype(np.uint8))
        axes[0].set_title("Source")
        axes[1].imshow(adapted_image.transpose((1, 2, 0)))
        axes[1].set_title("FDA_CD → Akoya")
        plt.suptitle(f"Fourier Domain Adaptation with white filter and amplitude distribution matching to Akoya L={L}")
        plt.savefig(os.path.join(output_folder, filename))
        plt.close(fig)
    
    if display:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(src_image.transpose((1, 2, 0)).astype(np.uint8))
        axes[0].set_title("Source")
        axes[1].imshow(adapted_image.transpose((1, 2, 0)))
        axes[1].set_title("FDA_CD → Akoya")
        plt.suptitle(f"Fourier Domain Adaptation with white filter and amplitude distribution matching to Akoya L={L}")
        plt.show()

    return adapted_image


## Main function
def FDA_CD(src_img, save=False, output_folder="", display=False):
    
    vector = Flat_log_Fourier(src_img)

    # reshape because transposition happened during creation of amplitudes dataset
    pred_label = int(classifier.predict(vector.reshape(1, -1))) 
    
    src_cdfs = distrib_KFBio[pred_label]
    tgt_cdfs = distrib_Akoya[pred_label]

    vector_KFBio_to_Akoya = match_vector_to_target_distribution(vector, src_cdfs, tgt_cdfs)

    return amplitude_swap(src_img, vector_KFBio_to_Akoya,0.01, save, output_folder, display)


# Examples
'''
list_train_scanners = ['Akoya', 'Leica']
input_root = '/home/leolr-int/nfs/data/data/patched/dim_256'
train_or_test = 'Train'
X_train, y_train, groups_train = load_data(input_root, train_or_test, L=0.01)
X_test, y_test, groups_test = load_data(input_root, 'Test', L=0.01)
path='/home/leolr-int/nfs/transformed_data/weights'
os.makedirs(f"{path}/{train_or_test}", exist_ok=True)
np.save(f'{path}/{train_or_test}/amplitudes.npy', X_train)
np.save(f'{path}/{train_or_test}/labels.npy', y_train)
np.save(f'{path}/{train_or_test}/groups.npy', groups_train)
akoya_id = int(X_test.shape[0]/2)
print(akoya_id)
X_train_akoya = X_train[:akoya_id,]
y_train_akoya = y_train[:akoya_id]
print(X_train_akoya.shape)
print(y_ttrain_akoya.shape)

distrib_Akoya = compute_smoothed_cdfs(X_train_akoya, y_ttrain_akoya)
save_path = "/home/leolr-int/nfs/transformed_data/weights/cdf_Akoya.pkl"
joblib.dump(distrib_Akoya, save_path)


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
'''
