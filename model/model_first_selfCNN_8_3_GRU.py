
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


class SelfConv(nn.Module):
    def __init__(self,  dim=256, nc=1,input_shape=[],n_classes=2, hidden_size=128, bidirectional=True,nlayers_Bigru=1, plot=False):
        super().__init__()
        self.plot=plot
        
        self.hidden_size=hidden_size
        self.bidirectional=bidirectional
        self.nlayers_Bigru=nlayers_Bigru
        self.n_classes=n_classes
        self.input_shape=input_shape

        self.conv1=nn.Conv2d(nc, 8, kernel_size=(1,3), stride=(1,1))
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1=nn.MaxPool2d((1, 2))
        
        
  
        
        

        
        self.attention_cnn=SelfAttention(8,F.leaky_relu)
        
        outConvShape1,outConvShape2=self._get_conv_output(self.input_shape)
        #Number of features
        #input_size = outConvShape1*outConvShape2 #Channels 16
        input_size=outConvShape1*8
        
        
        self.GRU= torch.nn.GRU(input_size, self.hidden_size,
                    bidirectional=self.bidirectional, 
                    num_layers=self.nlayers_Bigru,
                    batch_first=True) 

        
        idx_bi = 1
        if self.bidirectional:
            idx_bi = 2
        
        output_Gru = int((self.hidden_size*self.nlayers_Bigru*idx_bi))
        
        output_Gru = int((self.hidden_size*idx_bi))*50 
        

        
        #self.BatchNorm_Gru=torch.nn.BatchNorm1d(self.input_shape[2]) #Output
        self.BatchNorm_Gru=torch.nn.BatchNorm1d(50)
        
        
        
       
        #self.BatchNorm_fc=torch.nn.BatchNorm1d(256)
        self.linear=torch.nn.Linear(output_Gru,dim)
        self.linear_class=torch.nn.Linear(dim,self.n_classes)
        

        #self.conv4=nn.Conv2d(16, 32, kernel_size=(2,2), stride=(1,1),padding=(1,1), bias=False)
        #self.bn4 = nn.BatchNorm2d(32)
        
        #self.global_avg=nn.AdaptiveAvgPool2d((None ,1)) 
        #self.flatten=nn.Flatten()
        
        #self.linear = nn.Linear(16*13*16, dim)
        

        #torch.nn.init.kaiming_uniform_(self.conv1.weight)
        #torch.nn.init.kaiming_uniform_(self.conv2.weight)
        #torch.nn.init.kaiming_uniform_(self.conv_f.weight)
        #torch.nn.init.kaiming_uniform_(self.conv4.weight)
        #torch.nn.init.kaiming_uniform_(self.global_avg.weight)

    def _get_conv_output(self,shape):
    #        bs = 1 #batch size
        inputd = Variable(torch.rand(*shape))
        output_feat = self._forward_Conv(inputd)
        n_size = output_feat.size(3)#output_feat.data.view(inputd.size()[0], -1).size(1)
        n_size2 = output_feat.size(2)
        return n_size,n_size2
    
    def _forward_Conv(self, x):
        """        
        Convolutional layer features     
        ELU, max pooling and dropout
        """
        
        x_org=x
        if self.plot:
            plt.figure()
            for i in range(1,x_org.size()[1]+1):
                plt.subplot(1,3,i)
                img_conv = x_org[0,i-1,:,:].detach().cpu().numpy()
                
                plt.imshow(np.flipud(img_conv.T), cmap='gist_ncar', interpolation='bilinear')
                
                
            plt.tight_layout()
            plt.savefig('org_2ch.pdf')
        
        
        x =F.leaky_relu((self.bn1(self.pool1(self.conv1(x)))))
        
        if self.plot:
            plt.figure()
            for i in range(1,x.size()[1]+1):
                plt.subplot(2,4,i)
                img_conv = x[0,i-1,:,:].detach().cpu().numpy()
                
                plt.imshow(np.flipud(img_conv.T), cmap='gist_ncar', interpolation='bilinear')
                
                
            plt.tight_layout()
            plt.savefig('8ch_cnn1.pdf')
        
        
        
        
        x,attn=self.attention_cnn(x)
        
        if self.plot:
            plt.figure()
            for i in range(1,x.size()[1]+1):
                plt.subplot(4,4,i)
                img_conv = x[0,i-1,:,:].detach().cpu().numpy()
                
                plt.imshow(np.flipud(img_conv.T), cmap='gist_ncar', interpolation='bilinear')
                
                
            plt.tight_layout()
            plt.savefig('att_cnn1.pdf')
#        x = F.dropout2d(x)
        
        
#        x = F.dropout2d(x)
        return x  

    def _forward_Gru(self, x):
   
        out=x
        out = out.permute(0,2,1,3)#Permute dimensions to keep one-to-one context
        out = out.contiguous().view(out.shape[0],out.shape[1],-1)#Concatenate
        out,hiddens=self.GRU(out)
        #out=hiddens.permute(1,0,2)
        #out=self.GlobalAvgPool(out)
        out=self.BatchNorm_Gru(out)

        return out 
    
    def forward(self, x):
        
        
        x = self._forward_Conv(x)
        conv=x
        x=self._forward_Gru(x)
        
        x=F.gelu(self.linear(x.view(x.size(0), -1)))
        emb =x
        x=self.linear_class(x)
        
        
            
        #x =F.leaky_relu((self.bn4(self.pool(self.conv4(x)))))
        #x = x.view(x.size(0), -1)
        return x,emb 



class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//2 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//2 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out * x
        return out,attention



