#import typing
#typing.Self = typing.Any

import os
import warnings

warnings.filterwarnings("ignore")
import torch
import librosa
from Spectrum_Tensors_Mag import Spectrum_2D_Tensors
from normalizeFeatures import normalizeSpectrum
import numpy as np
from torch import nn
from model.model_first_selfCNN_8_3_GRU_lightning import SelfConv

def PADS_Models(file, plot=False, cuda=False, dims=['arousal', 'dominance', 'valence'], sig_set = False):
    device = 'cpu'
    if cuda:
        device = 'cuda'

    y, sr = librosa.load(file)

    audio = librosa.resample(y, orig_sr=sr, target_sr=16000)
    fs =16000

    #extract specs
    Spec_2D = Spectrum_2D_Tensors(audio, fs)
    specs=Spec_2D.get_2Dspectrogram_tensors()

    tensors_normalized = []
    for tensor in specs:
        tensor[0,:,:] = normalizeSpectrum(tensor[0,:,:])
        tensor[1,:,:] = normalizeSpectrum(tensor[1,:,:])
        tensor[2,:,:] = normalizeSpectrum(tensor[2,:,:])
        tensors_normalized.append(tensor)

    tensors_normalized = np.array(tensors_normalized)
    tensors_normalized=torch.from_numpy(tensors_normalized).to(device)

    predictions = {}
    for dim in dims:
        input_tensor_shape = (1,3,100,64)
        model = SelfConv.load_from_checkpoint('checkpoints/one_sec/'+dim+'_checkpoint.ckpt', input_shape=input_tensor_shape)
        model=model.to(device)
        model.eval()

        with torch.no_grad():
            scores, embeddings = model(tensors_normalized)
            print(dim + " scores range:", scores.min().item(), scores.max().item())
            class_prob = torch.softmax(scores, dim=1)
            predictions[dim+"_post"] = class_prob

    return predictions
