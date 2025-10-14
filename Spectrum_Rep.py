#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:18:26 2020
@author: paulaperezt
"""

from pyexpat import features
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
#import feat_ext as fext #Feature extraction
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from scipy.fftpack import dct

from tqdm import tqdm

def create_fold(new_folder):
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)



class Spectrum_Tensors:

    """
    It creates a spectrogram tensors with max 3 channels: (1) wideband spectrogram, (2) middle band, and (3) narrowband spectrogram
    :param sig: speech waveform
    :param fs: sampling frequency
    :param size1: window size for the first channel spectrogram (40 ms)
    :param size2: window size for the second channel spectrogram (16 ms)
    :param step: step size for both spectrograms (10ms)
    :param nfft1: frequency resolution of the first channel spectrogram (0-> int(2 ** np.ceil(np.log(win_size) / np.log(2))))
    :param nfft2: frequency resolution of the second channel spectrogram (0-> int(2 ** np.ceil(np.log(win_size) / np.log(2))))
    :param nfft3: frequency resolution of the second channel spectrogram (0-> int(2 ** np.ceil(np.log(win_size) / np.log(2))))
    :param seq_dim: tensor length (500ms)
    :param seq_step: step-size for the tensors (250ms) 
    """    
    
    def __init__(self, sig,fs, size1=0.015,size2=0.025,size3=0.045, step=0.01, nfft1=2048, nfft2=2048, nfft3=2048, seq_dim=1.0, seq_step=0.5):
        
        self.sig=sig
        self.size1 = size1
        self.size2 = size2
        self.size3 = size3
        self.step = step
        self.seq_dim = int(seq_dim/(step))
        self.seq_step=int(seq_step/(step))
        self.nfft=2048
        self.nfilters = 64 #Number of filters for MFCCs, GFCC,... len(nfft)/6
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
            Y = np.absolute(Y)
            
            # non-redundant part
            m = int(nfft / 2) + 1
            Y = Y[:, :m]

            Img = Y.imag
            Real = Y.real
            spec = np.sqrt(Real ** 2 + Img ** 2) #Magnitude

            return spec

        except Exception:
            print('No computed Spectrogram')
            return None
            pass

    def mel_spectrum(self,sig_spec, melfb):
        """
        sig_spec: STFT of the speech signal
        melfb: Mel filterbank - get_filterbanks()
        """
        spec_mel = np.dot(sig_spec, melfb.T)
        spec_mel = np.where(spec_mel == 0.0, np.finfo(float).eps, spec_mel)
        return np.log(spec_mel)

    def get_mfccs(self, spec, nceps=12, lifter=22):


        feat = np.dot(spec, self.melfb.T)  # compute the filterbank energies
        feat = np.where(feat == 0, np.finfo(float).eps, feat)  # if feat is zero, we get problems with log#
        feat = np.log(feat)
        feat = dct(feat, type=2, axis=1, norm='ortho')[:, :nceps]

        #Lifter

        if lifter > 0:
            nframes, ncoeff = np.shape(feat)
            n = np.arange(ncoeff)
            lift = 1 + (lifter / 2) * np.sin(np.pi * n / lifter)
            feat= lift * feat


        return feat


        
    def plot_2Dspectrogram(self, file_name):   
        """
        plot the 2-channel spectrogram for a speech signal
        :param file_name: file to save the image
        :returns: image with the 2-channel spectrogram
        """    

        spec1=self.get_spec(self.sig, size=self.size1, step=self.step, nfft_min=self.nfft1)
        spec2=self.get_spec(self.sig, size=self.size2, step=self.step, nfft_min=self.nfft2)
        spec3=self.get_spec(self.sig, size=self.size3, step=self.step, nfft_min=self.nfft3)
        
        spec1=mel.mel_spectrum(spec1,self.melfb)
        spec2=mel.mel_spectrum(spec2,self.melfb)
        spec3=mel.mel_spectrum(spec3,self.melfb)
        #spec2_nocut=spec2.copy()
        #spec2=spec2[:,:spec1.shape[1]]
        #spec1=spec1[:spec2.shape[0],:]
        
        spec1=np.flipud(spec1.T)
        spec2=np.flipud(spec2.T)
        spec3=np.flipud(spec3.T)

        # equivalent but more general
        
        plt.figure(figsize=(20,12))
        
        # add a polar subplot
        plt.subplot(311)
        plt.title('Window of 16 ms', fontsize=16)
        plt.imshow(spec1,interpolation='bilinear',
              aspect='auto',extent=[0,spec1.shape[1],0,spec1.shape[0]], cmap='inferno')
        
        plt.subplot(312)
        plt.title('Window of 25 ms', fontsize=16)
        plt.imshow(spec2,interpolation='bilinear',
              aspect='auto',extent=[0,spec2.shape[1],0,spec2.shape[0]], cmap='inferno')
        
        plt.subplot(313)
        plt.title('Window of 40 ms', fontsize=16)
        plt.imshow(spec3,interpolation='bilinear',
              aspect='auto',extent=[0,spec3.shape[1],0,spec3.shape[0]], cmap='inferno')
        #rect = [left, bottom, width, height]
        #plt.subplots_adjust(bottom=0.1, right=0.83, top=0.9)
        #cax = plt.axes([0.85, 0.1, 0.02, 0.8])
        #cbar=plt.colorbar(cax=cax)
        #cbar.set_ticks([])
        #cbar.set_label(label='Energy',size=16)
        plt.tight_layout()
        plt.savefig(file_name+'.pdf')
        plt.close('all')
        #plt.show()
        
        

    def get_3Dspectrogram_tensors(self,  file_name=''):

        """
        get the 1-channel spectrogram for a speech signal and save it as a torh tensor
        :param file_name: file to save the tensor
        :returns: torch tensors with the 2-channel spectrogram
        """    

        spec1=self.get_spec(self.sig, size=self.size1, step=self.step, nfft_min=self.nfft1)
        spec2=self.get_spec(self.sig, size=self.size2, step=self.step, nfft_min=self.nfft2)
        spec3=self.get_spec(self.sig, size=self.size3, step=self.step, nfft_min=self.nfft3)
        
        spec1=self.mel_spectrum(spec1,self.melfb)
        spec2=self.mel_spectrum(spec2,self.melfb)
        spec3=self.mel_spectrum(spec3,self.melfb)

        tensors = []

        ini = 0
        end = self.seq_dim
        seq_solp = self.seq_step
        i = 0


        while (end <= self.lsig):
            # Create two channel spectrograms
            data = np.zeros((3, self.seq_dim, spec3.shape[1]), dtype=np.float32)
            data[0, :, :] = spec1[ini:end, :]
            data[1, :, :] = spec2[ini:end, :]
            data[2, :, :] = spec3[ini:end, :]
            data = torch.from_numpy(data)
            tensors.append(data)
            i = i + 1
            ini = ini + seq_solp
            end = ini + self.seq_dim

        #data = {'15ms': spec1, '25ms': spec2, '45ms': spec3}

        return tensors


    def get_3DMFCCs(self, file_name='', nceps=40, lifter=84):

        """
        get the 3-channel MFCCs for a speech signal and save it as a npy file
        :param file_name: file to save the tensor
        :returns: torch tensors with the 3-channel spectrogram
        """

        spec1=self.get_spec(self.sig, size=self.size1, step=self.step, nfft_min=self.nfft1)
        spec2=self.get_spec(self.sig, size=self.size2, step=self.step, nfft_min=self.nfft2)
        spec3=self.get_spec(self.sig, size=self.size3, step=self.step, nfft_min=self.nfft3)

        #Get MFCCs
        mfccs1=self.get_mfccs(spec1, nceps=nceps, lifter=lifter)
        mfccs2=self.get_mfccs(spec2, nceps=nceps, lifter=lifter)
        mfccs3=self.get_mfccs(spec3, nceps=nceps, lifter=lifter)



        data = {'15ms':mfccs1, '25ms': mfccs2, '45ms': mfccs3}



        return data

def getFeaturesTensors(audio_path, feats_path):
    size1=float(0.015)
    size2=float(0.025)
    size3=float(0.045)
    step=float(0.01)
    nfft1=int(2048)
    nfft2=int(2048)
    nfft3=int(2048)
    seq_dim=float(1.0)
    seq_step=float(0.5)

    all_files=os.listdir(audio_path)
    print("Audio files found in given save_path:", all_files)
    folders =[]
    features = {}
    for f in all_files:
        if not os.path.isfile(f):
            folders.append(f)

    print("Folders to be processed:", folders)

    for folder in folders:
        print("Processing audio files in:", folder)
        files= np.hstack(sorted([".".join(f.split(".")[:-1]) for f in os.listdir(audio_path+'/'+folder) ]))
        print(files)
        
        pbar=tqdm(files)

        create_fold(feats_path+'/'+folder)
        feats_tensors = {}
        for file in pbar:
            pbar.set_description("Processing %s" % file)
            #fs,sig = read(audio_path+'/'+folder+'/questionnaire_audio/'+file+'.wav')
            try:
                y, sr = librosa.load(audio_path+'/'+folder+'/'+file+'.wav')

                sig = librosa.resample(y, orig_sr=sr, target_sr=16000)
                fs=16000

                Spec=Spectrum_Tensors(sig,fs, size1, size2,size3, step, nfft1, nfft2,nfft3, seq_dim, seq_step)



                feats_tensors[file] =   Spec.get_3Dspectrogram_tensors(file)
            #except:
            except Exception as e:
                print(e)

        features[folder] = feats_tensors
        features_folder = feats_path+'/'+folder
        if not os.path.exists(features_folder):
            os.makedirs(features_folder)
        print("Saving features in folder:", feats_path + folder)
        np.save(os.path.join(features_folder, str(folder) + "_feats"), feats_tensors,allow_pickle=True)

    
    return features, feats_path
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('audio_path', default= r"./audios/",help='Audio file folder to compute the spectrograms and mfccs') #
    parser.add_argument('save_path', default='./feats/', help='Path to save the tensors and the figs') #
    parser.add_argument('--size1', default='0.015',help='Window size of the first spectrogram in seconds, must be shorter than the second spectrogram. By default 15 ms.')
    parser.add_argument('--size2', default='0.025',help='Window size of the second spectrogram in seconds, must be longer than the first spectrogram. By default 25 ms.')
    parser.add_argument('--size3', default='0.045',help='Window size of the second spectrogram in seconds, must be longer than the first spectrogram. By default 45 ms.')

    parser.add_argument('--step', default='0.01', help='Step size in seconds. By default 10 ms.')
    parser.add_argument('--nfft1', default='2048', help='Fourier transform resolution for the first spectrogram. By default 0, because later is computed.')
    parser.add_argument('--nfft2', default='2048', help='Fourier transforfor the second spectrogramm resolution for the second spectrogram. By default 0, because later is computed.')
    parser.add_argument('--nfft3', default='2048', help='Fourier transforfor the second spectrogramm resolution for the second spectrogram. By default 0, because later is computed.')

    parser.add_argument('--seq_dim', default='1.0', help='Sequence dimension in seconds . By default 1.0.')
    parser.add_argument('--seq_step', default='0.5', help='Sequence step dimension in seconds . By default 0.5.')

    args = parser.parse_args()

    audio_path=args.audio_path 
    save_path=args.save_path 
    size1=float(args.size1)
    size2=float(args.size2)
    size3=float(args.size3)
    step=float(args.step)
    nfft1=int(args.nfft1)
    nfft2=int(args.nfft2)
    nfft3=int(args.nfft3)
    seq_dim=float(args.seq_dim)
    seq_step=float(args.seq_step)


    all_files=os.listdir(audio_path)
    print("Audio files found in given save_path:", all_files)
    folders =[]
    for f in all_files:
        if not os.path.isfile(f):
            folders.append(f)

    print("Folders to be processed:", folders)

    for folder in folders:
        print("Processing audio files in:", folder)
        files= np.hstack(sorted([".".join(f.split(".")[:-1]) for f in os.listdir(audio_path+'/'+folder) ]))
        print(files)
        
        pbar=tqdm(files)

        create_fold(save_path+'/'+folder)
        feats_mfccs={}
        feats_specs = {}
        for file in pbar:
            pbar.set_description("Processing %s" % file)
            #fs,sig = read(audio_path+'/'+folder+'/questionnaire_audio/'+file+'.wav')
            try:
                y, sr = librosa.load(audio_path+'/'+folder+'/'+file+'.wav')

                sig = librosa.resample(y, orig_sr=sr, target_sr=16000)
                fs=16000

                Spec=Spectrum_Tensors(sig,fs, size1, size2,size3, step, nfft1, nfft2,nfft3, seq_dim, seq_step)



                feats_specs[file] =   Spec.get_3Dspectrogram_tensors(file)
                #feats_mfccs[file] = Spec.get_3DMFCCs()
            #except:
            except Exception as e:
                print(e)

        features_folder = save_path+'/'+folder
        if not os.path.exists(features_folder):
            os.makedirs(features_folder)
        print("Saving features in folder:", features_folder)
        np.save(os.path.join(features_folder, str(folder) + "_feats"), feats_specs,allow_pickle=True)
 
