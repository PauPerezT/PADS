
"""
Created on Sun Jan 30 15:45:14 2022
@author: Paula Perez
"""

import os
import warnings

warnings.filterwarnings("ignore")
import torch
import json
import requests
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import gc
import librosa

from Spectrum_Tensors_Mag import Spectrum_2D_Tensors


from tqdm import tqdm
from scipy.io.wavfile import write, read
import numpy as np

torch.cuda.empty_cache()
gc.collect()

import soundfile as sf
def create_fold(new_folder):
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
from model.model_first_selfCNN_8_3_GRU import  SelfConv
def tensor_min_max(input_tensor,norm_c1,norm_c2,norm_c3):
    tensor = input_tensor
    tensor[0,:,:] = (input_tensor[0,:,:]-np.float(norm_c1['min']))/(np.float(norm_c1['max'])-np.float(norm_c1['min']))
    tensor[1,:,:] = (input_tensor[1,:,:]-np.float(norm_c2['min']))/(np.float(norm_c2['max'])-np.float(norm_c2['min']))
    tensor[2,:,:] = (input_tensor[2,:,:]-np.float(norm_c3['min']))/(np.float(norm_c3['max'])-np.float(norm_c3['min']))

    return tensor
# %%
def PADS_Models(file, plot=False, cuda=False, dims=['arousal', 'valence', 'dominance'], sig_set = False):
    path = os.path.dirname(os.path.abspath(__file__)) + '/'

    # --------------------------------------------------------------------------
    with open(path + 'data_norm/param_c1.json', 'r') as fp:
        norm_c1 = json.load(fp)
        # --------------------------------------------------------------------------
    with open(path + 'data_norm/param_c2.json', 'r') as fp:
        norm_c2 = json.load(fp)
        # --------------------------------------------------------------------------
    with open(path + 'data_norm/param_c3.json', 'r') as fp:
        norm_c3 = json.load(fp)




    y, sr = librosa.load(file)

    audio = librosa.resample(y, orig_sr=sr, target_sr=16000)
    fs =16000
    device = 'cpu'
    if cuda:
        device = 'cuda'


    #extract specs
    Spec_2D = Spectrum_2D_Tensors(audio, fs)
    specs=Spec_2D.get_2Dspectrogram_tensors()
    specs=tensor_min_max(specs,norm_c1,norm_c2,norm_c3)
    specs=torch.from_numpy(specs).to(device)
    #

    outputs = {}
    gc.collect()
    torch.cuda.empty_cache()
    for dim in dims:
        input_shape = (1,3,50,64)
        model = SelfConv(nc=3,input_shape=input_shape)
        ckp = torch.load('checkpoints/'+dim+'_checkpoint.ckp', device)
        model.load_state_dict(ckp['state_dict'])

        model=model.to(device)





        with torch.inference_mode():

            scores = []
            emb = []

            iter = len(specs) // 5 #TODO batch sampler
            if iter !=0:
                for i in range(iter):

                    if i != iter - 1 and len(specs) > 5:

                        score, em = model(specs[i * 5:(i + 1) * 5])

                    elif len(specs) <= 5:
                        score, em = model(specs)

                    else:

                        score, em = model(specs[i * 5:])


                    if sig_set:
                        score = torch.nn.functional.sigmoid(score)
                    scores.append(score.cpu())
                    emb.append(em.cpu())
            else:
                score, em = model(specs)
                if sig_set:
                    score = torch.nn.functional.sigmoid(score)
                scores.append(score.cpu())
                emb.append(em.cpu())

            scores = np.vstack(scores)
            emb = np.vstack(emb)


            outputs[dim+'_embs']= emb
            outputs[dim+'_post'] = scores

        gc.collect()
        torch.cuda.empty_cache()




    return outputs








# # %%

gc.collect()
torch.cuda.empty_cache()

save_path= r'./feats/'
audio_path = r'./audios/'




files= np.hstack(sorted([".".join(f.split(".")[:-1]) for f in os.listdir(audio_path) ]))

if len(files)>0 :







    feats = {}

    pbar = tqdm(files)

    for file in pbar:

        pbar.set_description("Processing %s" % file)

        try:


            pt = audio_path+file+'.wav'

            feats[file] = PADS_Models(pt, cuda=True, sig_set = False)

            torch.cuda.empty_cache()
            gc.collect()
        except Exception as error:
            print("An exception occurred:", error, file)
            pass

        torch.cuda.empty_cache()
        gc.collect()
    create_fold(save_path)
    np.save(save_path +'/pad.npy', feats_specs, allow_pickle=True)








gc.collect()
