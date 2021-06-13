#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 20:49:04 2021

@author: P.A. Perez-Toro
"""

from scipy.io.wavfile import read,write #Leer y guardar audios
import numpy as np
import os,sys
from matplotlib import ticker
import matplotlib.pyplot as plt
import json
import argparse
import pickle
import mel_extractor as mel #Feature extraction
import librosa
import sigproc as sg  
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from tqdm import tqdm

def create_fold(new_folder):
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)



class Spectrum_3D_Tensors:

    """
    It creates a spectrogram tensors with 3 channels to be used by the EmoSSpeech: (1) 16 ms window, (2) 25 ms window, and (3) 40 ms window
    :param sig: speech waveform
    :param fs: sampling frequency
    :param size1: window size for the first channel spectrogram (16 ms)
    :param size2: window size for the second channel spectrogram (25 ms)
    :param size2: window size for the second channel spectrogram (45 ms)
    :param step: step size for both spectrograms (10ms)
    :param nfft1: frequency resolution of the first channel spectrogram 
    :param nfft2: frequency resolution of the second channel spectrogram 
    :param nfft3: frequency resolution of the second channel spectrogram
    :param seq_dim: tensor length (500ms)
    :param seq_step: step-size for the tensors (250ms) 
    """    
    
    def __init__(self, sig,fs, size1=0.016,size2=0.025,size3=0.040, step=0.01, nfft1=2048, nfft2=2048, nfft3=2048, seq_dim=0.5, seq_step=0.5/2): 
        
        self.sig=sig
        self.size1 = size1
        self.size2 = size2
        self.size3 = size3
        self.step = step
        self.seq_dim = int(seq_dim/(step))
        self.seq_step=int(seq_step/(step))
        self.nfilters = 64 #Number of filters for MFCCs
        self.fmax = 8000 #Maximum sampling frequency to compute filter banks
        #Mel filter bank
        self.melfb = mel.get_filterbanks(self.nfilters,nfft = self.nfft,samplerate = self.fmax)
        self.fs = fs
        self.nfft1=nfft1
        self.nfft2=nfft2
        self.nfft3=nfft3

        
    def get_spec(self, sig, size=0.04, step=0.01,nfft_min=0):
        """
        Compute a spectrogram
        :param sig: speech waveform
        :param fs: sampling frequency
        :param size: window size (40 ms)
        :param step: step size (10ms)
        :param nfft_min: frequency resolution (0-> int(2 ** np.ceil(np.log(win_size) / np.log(2))))
        :returns: spectrogram
        """    

        win_size = int(size*self.fs)
        step_size = int(step*self.fs)
        
        try:
            #Normalization
            
            sig = sig-np.mean(sig)
            sig = sig/np.max(np.abs(sig))
            
            #Framing
            n_frames = int((len(sig) - win_size) / step_size)
            
            # extract frames
            windows = [sig[i * step_size : i * step_size + win_size] 
                       for i in range(n_frames)]
 
            frames=np.vstack(windows)
                                    
            #Hamming
            frames *= np.hamming(win_size)
            self.lsig = frames.shape[0]       
            #Power spectrum with FFT. 
            
            # zero padding to next power of 2
            if nfft_min==0:
                nfft = int(2 ** np.ceil(np.log(win_size) / np.log(2)))
            else:
                nfft = nfft_min
            # Fourier transform


            Y = np.fft.fft(frames, n=nfft)
            #Y = np.absolute(Y)
            
            # non-redundant part
            m = int(nfft / 2) + 1
            Y = Y[:, :m]
            Img=Y.imag
            Real=Y.real
            
            #Magnitude
            spec= np.sqrt(Real**2+ Img**2)
            
            
            return spec

        except Exception:
            print('No computed Spectrogram')
            return None
            pass


    
    def get_spectrogram_tensors(self):

        """
        get the 3-channel spectrogram for a speech signal and save it as a torh tensor
        :returns: torch tensors with the 3-channel spectrogram
        """    

        spec1=self.get_spec(self.sig, size=self.size1, step=self.step, nfft_min=self.nfft1)
        spec2=self.get_spec(self.sig, size=self.size2, step=self.step, nfft_min=self.nfft2)
        spec3=self.get_spec(self.sig, size=self.size3, step=self.step, nfft_min=self.nfft3)
        
        spec1=mel.mel_spectrum(spec1,self.melfb)
        spec2=mel.mel_spectrum(spec2,self.melfb)
        spec3=mel.mel_spectrum(spec3,self.melfb)
        tensors=[]        

        ini = 0
        end = self.seq_dim
        seq_solp = self.seq_step
        i = 0
        while(end<=self.lsig):
            #Create three channel spectrograms
            data = np.zeros((3,self.seq_dim,spec1.shape[1]),dtype = np.float32)
            data[0,:,:] = spec1[ini:end,:]
            data[1,:,:] = spec2[ini:end,:]
            data[2,:,:] = spec3[ini:end,:]
            data = torch.from_numpy(data)
            tensors.append(data)
            i = i+1
            ini = ini+seq_solp
            end = ini+self.seq_dim
        return tensors

