import numpy as np 
import os
import torch as t
from Spectrum_Rep import getFeaturesTensors


# Normalize tensors using min and max spectral components
def normalizeFeaturesTensors(features):
    all_files_tensors_norm = {}
    for folder_name, all_files_tensors  in features.items():
        file_tensors_norm = {}
        for file_name, file_tensors in all_files_tensors.items():
            tensors_list_norm = []
            for tensor in file_tensors:
                tensor[0,:,:] = normalizeSpectrum(tensor[0,:,:])
                tensor[1,:,:] = normalizeSpectrum(tensor[1,:,:])
                tensor[2,:,:] = normalizeSpectrum(tensor[2,:,:])
                tensors_list_norm.append(tensor)
            file_tensors_norm[file_name] = tensors_list_norm
        all_files_tensors_norm[folder_name] = file_tensors_norm

    normalized_features = all_files_tensors_norm
    return normalized_features

# Normalize spectrum using min and max spectral components
def normalizeSpectrum(spectrum):
    time_idx = 0
    for freq_bands in spectrum:
        spectrum[time_idx,:] = freq_bands-np.float32(freq_bands.min())/(np.float32(freq_bands.max())-np.float32(freq_bands.min()))
        time_idx += 1

    return spectrum

if __name__ == '__main__':
    features = getFeaturesTensors()
    feats_normalized = normalizeFeaturesTensors(features)
