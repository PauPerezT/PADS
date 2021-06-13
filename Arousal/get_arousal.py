#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speech 2D Spectrograms 

->train.py
Created on Sat May  9 15:32:48 2020
@author: P.A. Perez-Toro
@email:paula.perezt@udea.edu.co
"""


import torch as t
import torchvision as tv
from data import get_dataset
from stopping import EarlyStoppingCallback
from trainer import Trainer
from Tester import Tester
from matplotlib import pyplot as plt
import numpy as np
from model.model import  SelfConv
import warnings
import json
import pandas as pd
from utils import create_fold
import csv
import os
import argparse
import librosa
from tqdm.autonotebook import tqdm
from scipy.io.wavfile import read,write #read, write audios

#warnings.filterwarnings("ignore")
# set up data loading for the training and validation set using t.utils.data.DataLoader and the methods implemented in data.py

def compute(audio, file_name,save_path='./', bs=1, cuda=False):
    
    if cuda:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
    else:
        device="cpu"
    

    
    if bs==1:
        params = {'batch_size': bs,
                  'shuffle': False,
                  'drop_last':False}
    else:
        params = {'batch_size': bs,
              'shuffle': False,
              'drop_last':True,
              'pin_memory':True}
    
    
    
    path=os.path.dirname(os.path.abspath(__file__))+'/'
    
    #--------------------------------------------------------------------------   
    with open(path+'data_norm/param_c1.json', 'r') as fp:
        norm_c1= json.load(fp)  
    #--------------------------------------------------------------------------   
    with open(path+'data_norm/param_c2.json', 'r') as fp:
        norm_c2= json.load(fp)  
    #--------------------------------------------------------------------------   
    with open(path+'data_norm/param_c3.json', 'r') as fp:
        norm_c3= json.load(fp)  
    
    
    
    
    data = get_dataset(audio, norm_c1,norm_c2,norm_c3)
    tensor = t.utils.data.DataLoader(data, **params)

    
    
     
    input_shape = (bs,3,50,64)
    
    # set up your model
    model=SelfConv(nc=3,input_shape=input_shape)
    
    print(50*'=','\n Model \n', model, '\n',50*'=')
    
    crit = t.nn.CrossEntropyLoss()
    labelstf=[]
    labelstf.append('File')   
    for n in range (256):
        labelstf.append('Neuron'+str(n+1)) 
    
    with open(save_path+'/Emb/Arousal/'+file_name+'_Emb.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
    
        writer.writerow(labelstf)
    
    # set up optimizer (see t.optim); 
    lr = 0.0001 
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    EarlyStopping=EarlyStoppingCallback(20)
    
    
    test=Tester(model,crit,optimizer,[], tensor, cuda = True, early_stopping_cb =EarlyStopping)
    
    test.test(file_name)
    

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('audio_path', default='./audios/',help='Audio file folder to compute the spectrograms')
    parser.add_argument('save_path', default='./',help='Path to save the tensors and the figs')
    parser.add_argument('--bs', default='1' ,help='Batch size for the computation of the different segments')
    parser.add_argument('--cuda', default='False',help='Cuda True or cuda False')


    args = parser.parse_args()

    audio_path=args.audio_path 
    save_path=args.save_path 
    bs=int(args.bs)
    cuda=args.cuda
    
    if cuda=='True':
        cuda=True
    else:
        cuda=False
    
    files= np.hstack(sorted([".".join(f.split(".")[:-1]) for f in os.listdir(audio_path) ]))

    pbar=tqdm(files)
    
    create_fold(save_path+'/Emb/Arousal/')
    create_fold(save_path+'/Post/Arousal/')
    for file in pbar:
        pbar.set_description("Processing %s" % file)
        fs,sig = read(audio_path+'/'+file+'.wav')
        
        if fs!=16000:
            y, sr = librosa.load(audio_path+'/'+file+'.wav')
            sig = librosa.resample(y, sr, 16000)
            fs=16000
            
        compute(sig, file,save_path='./', bs=bs, cuda=cuda)




