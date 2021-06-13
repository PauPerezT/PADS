
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Learning Exercise 4

->trainer.py

Created on Wed Feb 19 18:53:05 2020

@author: Paula Andrea Perez Toro
@email:paula.perezt@udea.edu.co
"""

import torch as t
from sklearn.metrics import f1_score,precision_recall_fscore_support, accuracy_score, confusion_matrix
from tqdm.autonotebook import tqdm
import numpy as np
import torchvision as tv
import time
import os
import warnings
import pandas as pd
import csv

warnings.filterwarnings("ignore")

t.cuda.empty_cache()
device = t.device("cuda" if t.cuda.is_available() else "cpu")
n_gpu = t.cuda.device_count()
#t.cuda.get_device_name(0)


class Tester:
    
    def __init__(self,               
                 model,                # Model to be trained.
                 crit,                 # Loss function
                 optim = None,         # Optimiser
                 train_dl = None,      # Training data set
                 dev_dl = None,   # Development (or test) data set
                 cuda = True,          # Whether to use the GPU
                 early_stopping_cb = None, files=[]): # The stopping criterion. 
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._dev_dl = dev_dl
        self._cuda = cuda
        self._early_stopping_cb = early_stopping_cb
        self.files=files
        
        self.batch_size=0
        
        self.epoch=0
        
        self.device="cpu"
        if self._cuda:
            self.device = t.device('cuda' if t.cuda.is_available() else "cpu")

            #self._model = model.cuda()
            #self._crit = crit.cuda()
            self._model = model.to(self.device)
            self._crit = crit.to(self.device)


    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/arousal.ckp', 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
            
    
    def dev_step(self, x):
        
        # predict
        
        # propagate through the network and calculate the loss and predictions
        scores,emb = self._model(x)
        #===================================================

        #Compute losses and accuracies per batch
        activation=t.nn.Sigmoid()
        posteriors=activation(scores.data).cpu().detach().numpy()
        y_pred=t.max(activation(scores.data),1)[1]
        #y_pred=t.max(scores.data,1)[1]
        #y_pred = y_pred.cpu().detach().numpy()
        #y_pred = ypred
        #=========================================================

                   
        # return the loss and the predictions
        return  y_pred,posteriors,emb
      

    
    def extract(self,file):
        
        # set eval mode
        self._model.eval() # prep model for evaluation

        y_pred_dev=[]
        loss_dev=[]
        posteriors=[]
        y=[]
 
        
        # disable gradient computation
        with t.no_grad():
            i=0
            # iterate through the development set
            for idx_batch2,(test_inputs) in enumerate(self._dev_dl,1):
            #for (test_inputs, test_labels) in enumerate(self._dev_dl,1):
                # transfer the batch to the gpu if given
                test_inputs = test_inputs.to(self.device, dtype=t.float)

                
                print('Batch: {}/{}'.format(idx_batch2,len(self._dev_dl)),end='\r')

                # perform a development step
                y_pred,post, emb=Tester.dev_step(self,test_inputs)
                
                emb=emb.cpu().detach().numpy()
                
                with open(save_path+'/Emb/Arousal/'+file+'_Emb.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerows(np.hstack((np.vstack(self.files[i:i+len(emb)]), emb))) 
                posteriors.append(post)
                #embeddings.append(emb.cpu().detach().numpy())
                # save the predictions and the labels for each batch

                y_pred_dev.append(y_pred.cpu().detach().numpy())
                i+=len(emb)
            
            
            
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        #Mean losses
        

        

        yp = np.hstack(y_pred_dev)
        posteriors=np.vstack(posteriors)

        return yp,posteriors
        
    

        
    
    def test(self,file,save_path):
        starting_point = time.time()
        
        # create a list for the train and validation losses, and create a counter for the epoch 

        
        self.epoch_n=0
        self.batch_size=self._dev_dl.batch_size
        
        self.restore_checkpoint( self.epoch)


        y_pred,post=self.extract(file,save_path)
        
        #Storing Embeddings


        
        # use the save_checkpoint function to save the model for each epoch

        

        newfile=[]
        for f in files:
            newfile.append(f)
            
        newfile=np.array(newfile)
        database = {'File Name': files,
        'Predicted Labels': y_pred,
        'High Arousal': post[:,0],
        'Low Arousal': post[:,1]
        }
        df = pd.DataFrame(database)
        
        
        
        
        df.to_csv (save_path+'/Post/Arousal/'+file+'_post.csv', index = False, header=True) #Don't forget to add '.csv' at the end of the path


        
        # check whether early stopping should be performed using the early stopping callback and stop if so
        
        elapsed_time = time.time () - starting_point
        elapsed_time = np.round(elapsed_time/60,2) 
        print('Epoch: {} Time Duration: {}'.format(self.epoch, elapsed_time))
        #Increasing counter
        
        
        
    
        

        
 