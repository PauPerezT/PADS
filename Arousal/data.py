#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IEMOCAP Speech 2D Spectrograms
Quadrant

->data.py

Created on Sat May  9 15:32:48 2020

@author: P.A. Perez-Toro
@email:paula.perezt@udea.edu.co
"""

import torch
from torch.utils import data
from torch.utils.data import Dataset
import os
import numpy as np 
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from scipy.io.wavfile import read,write #read and write audios
from sklearn.utils import class_weight
from skimage.transform import resize
from Spectrum import Spectrum_3D_Tensors
import torchvision as tv
import random
from itertools import chain
import librosa
#%%



class SpeechDataset(Dataset):
    
    def __init__(self, files, norm_c1, norm_c2,norm_c3):
        

        

        self.norm_c1=norm_c1
        self.norm_c2=norm_c2
        self.norm_c3=norm_c3

        
        self.files=files
        

    
    

        
    def __getitem__(self, index):
        
        if torch.is_tensor(index):
            index = index.tolist()
        
        input_tensor = torch.Tensor(self.files[index])
        
        
        #Min Max Normalization
        input_tensor=tensor_min_max(input_tensor,self.norm_c1,self.norm_c2,self.norm_c3)
        
        #spec=spec.unsqueeze(0)
        
        

        
        return (input_tensor)

    def __len__(self):
        
        
        
        return len(self.files)
    
    

def tensor_min_max(input_tensor,norm_c1,norm_c2,norm_c3):
    tensor = input_tensor
    tensor[0,:,:] = (input_tensor[0,:,:]-np.float(norm_c1['min']))/(np.float(norm_c1['max'])-np.float(norm_c1['min']))
    tensor[1,:,:] = (input_tensor[1,:,:]-np.float(norm_c2['min']))/(np.float(norm_c2['max'])-np.float(norm_c2['min']))
    tensor[2,:,:] = (input_tensor[2,:,:]-np.float(norm_c3['min']))/(np.float(norm_c3['max'])-np.float(norm_c3['min']))

    return tensor        
        



def get_dataset(sig, norm_c1,norm_c2,norm_c3):
    #print('%-----Test tensors-----%')
    
    


    
    spec=Spectrum_3D_Tensors(sig,16000)
    tensors=spec.get_spectrogram_tensors()
    
    speechData=SpeechDataset(tensors,norm_c1, norm_c2,norm_c3)

    

    return speechData
