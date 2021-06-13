# This file includes routines for basic signal processing including framing and computing power spectra.
# Author: James Lyons 2012

import os
import numpy as np
import math
from scipy.io.wavfile import read #Leer y guardar audios
import scipy as sp
from scipy.signal import hilbert, gaussian
from scipy.signal import firwin,lfilter

def check_audio(audio_path):
    """
    Obtener frecuencia de muestreo y numero de canales
    Entrada
        :param audio_path: Carpeta que contiene los audios
    Salida
        :returns Frecuencia de muestreo promedio en la carpeta de audios
    """
    file_list = os.listdir(audio_path)
    list_fs = []
    for audio_name in file_list:
        fs,sig = read(audio_path+'/'+audio_name)
        list_fs.append(fs)
        channels = len(sig.shape)
        print('Audio: '+audio_name+' Fs: '+str(fs)+' Canales: '+str(channels))
    print('Fs Maximo: '+str(np.max(list_fs)))
    print('Fs Minimo: '+str(np.min(list_fs)))
    print('Fs promedio: '+str(np.mean(list_fs)))
    return np.mean(list_fs)

def norm_sig(sig):
    """Remove DC level and scale signal between -1 and 1.

    :param sig: Signal to normalize
    :returns: Normalized signal
    """
    #Eliminar nivel DC
    normsig = sig-np.mean(sig)
    #Escalar valores de amplitud entre -1 y 1
    normsig = normsig/float(np.max(np.absolute(normsig)))
    return normsig

def sig_contour(cont_list,sig,fs,win=0.04,solp=0.01):
     """Get contorus to plot along the speech signal
     :param cont_list: List with pitch values, energy,.....
     :param sig: Speech signal
     :param fs: sampling frequency
     :param win: win length in miliseconds
     :param solp: step size in miliseconds
     :returns: contour
     """
     cont = np.zeros(len(sig))
     siz = int(fs*solp)
     ini = 0
     end = siz+ini
     for i in cont_list:
          cont[ini:end] = i
          ini = end
          end = end+siz
#     g = float(max(np.absolute(min(sig)),max(sig)))#Valor maximo para escalar energía
#     cont = (g*cont)/float(max(np.absolute(min(cont)),max(cont)))
     return cont

def framesig(sig,frame_len,frame_step,winfunc=lambda x:np.ones((1,x))):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.    
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round(frame_len))
    frame_step = int(round(frame_step))
    if slen <= frame_len: 
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0*slen - frame_len)/frame_step))
        
    padlen = int((numframes-1)*frame_step + frame_len)
    
    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig,zeros))
    
    indices = np.tile(np.arange(0,frame_len),(numframes,1)) + np.tile(np.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = np.array(indices,dtype=np.int32)
    frames = padsignal[indices]
    win = np.tile(winfunc(frame_len),(numframes,1))
    return frames*win
    
    
def deframesig(frames,siglen,frame_len,frame_step,winfunc=lambda x:np.ones((1,x))):
    """Does overlap-add procedure to undo the action of framesig. 

    :param frames: the array of frames.
    :param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.    
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.    
    :returns: a 1-D signal.
    """
    frame_len = round(frame_len)
    frame_step = round(frame_step)
    numframes = np.shape(frames)[0]
    assert np.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'
 
    indices = np.tile(np.arange(0,frame_len),(numframes,1)) + np.tile(np.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = np.array(indices,dtype=np.int32)
    padlen = (numframes-1)*frame_step + frame_len   
    
    if siglen <= 0: siglen = padlen
    
    rec_signal = np.zeros((1,padlen))
    window_correction = np.zeros((1,padlen))
    win = winfunc(frame_len)
    
    for i in range(0,numframes):
        window_correction[indices[i,:]] = window_correction[indices[i,:]] + win + 1e-15 #add a little bit so it is never zero
        rec_signal[indices[i,:]] = rec_signal[indices[i,:]] + frames[i,:]
        
    rec_signal = rec_signal/window_correction
    return rec_signal[0:siglen]

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def magspec(frames,NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT. 

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the magnitude spectrum of the corresponding frame.
    """    
    complex_spec = np.fft.rfft(frames,NFFT)
    return np.absolute(complex_spec)
          
def powspec(frames,NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT. 

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the power spectrum of the corresponding frame.
    """    
    return 1.0/NFFT * np.square(magspec(frames,NFFT))
    
def logpowspec(frames,NFFT,norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT. 

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 1.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the log power spectrum of the corresponding frame.
    """    
    ps = powspec(frames,NFFT);
    ps[ps<=1e-30] = 1e-30
    lps = 10*np.log10(ps)
    if norm:
        return lps - np.max(lps)
    else:
        return lps
    
def preemphasis(signal,coeff=0.95):
    """perform preemphasis on the input signal.
    
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

def highpass_fir(sigR,fs,fc,nfil):
    #sigR: Sennal a filtrar
    #fs: Frecuencia de muestreo de la sennal a filtrar
    #fc: Frecuencia de corte.
    #nfil: Orden del filtro
    largo = nfil+1 #  orden del filtro
    fcN = float(fc)/(float(fs)*0.5) # Frecuencia de corte normalizada
    #Filtro pasa bajas
    h = firwin(largo, fcN)
    #Inversion espectral para obtener pasa altas    
    h = -h
    h[int(largo/2)] = h[int(largo/2)] + 1
    #Aplicar transformada
    sigF = lfilter(h, 1,sigR)
    return sigF

#*****************************************************************************
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
#*****************************************************************************
def extract_windows(signal, size, step):
    # make sure we have a mono signal
    assert(signal.ndim == 1)
    
#    # subtract DC (also converting to floating point)
#    signal = signal - signal.mean()
    
    n_frames = int((len(signal) - size) / step)
    
    # extract frames
    windows = [signal[i * step : i * step + size] 
               for i in range(n_frames)]
    
    # stack (each row is a window)
    return np.vstack(windows)
#*****************************************************************************
def powerspec(X, rate, win_duration, n_padded_min=0):
    win_size = int(rate * win_duration)
    
    # apply hanning window
    X *= np.hanning(win_size)
    
    # zero padding to next power of 2
    if n_padded_min==0:
        n_padded = max(n_padded_min, int(2 ** np.ceil(np.log(win_size) / np.log(2))))
    else:
        n_padded = n_padded_min
    # Fourier transform
#    Y = np.fft.rfft(X, n=n_padded)
    Y = np.fft.fft(X, n=n_padded)
    Y = np.absolute(Y)
    
    # non-redundant part
    m = int(n_padded / 2) + 1
    Y = Y[:, :m]
    
    return np.abs(Y) ** 2, n_padded
#*****************************************************************************
def powerspec2D(X, rate, win_duration, n_padded_min=0):
    win_size = int(rate * win_duration)
    
    # apply hanning window
    X *= np.hanning(win_size)
    
    # zero padding to next power of 2
    if n_padded_min==0:
        n_padded = max(n_padded_min, int(2 ** np.ceil(np.log(win_size) / np.log(2))))
    else:
        n_padded = n_padded_min
    # Fourier transform
#    Y = np.fft.rfft(X, n=n_padded)
    Y = np.fft.fft(X, n=n_padded)
#    Y = np.absolute(Y)
    
    # non-redundant part
    m = int(n_padded / 2) + 1
    Y = Y[:, :m]
    
    return Y, n_padded
#********************************************************
def read_file(file_name):
    """
    Converts the text in a txt, txtgrid,... into a python list
    """
    f = open(file_name,'r')
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n','')
    return lines
#****************************************************************************
def get_file(path,cond):
    """
    path: Folder containing the file name
    cond: Name, code, or number contained in the filename to be found:
          If the file name is 0001_ADCFGT.wav, cond could be 0001 or ADCFGT.
    """
    list_files = os.listdir(path)
    for f in list_files:
        if f.upper().find(cond.upper())!=-1:
            break
    if f.upper().find(cond.upper())==-1:
        f = ''#If no file is found, then return blank
    return f
#########################################
def hilb_tr(signal,fs,smooth=True,glen = 0.01):
    """
    Apply hilbert transform over the signal to get
    the envelop and time fine structure
    
    If smooth true, then the amplitude envelope is smoothed with a gaussian window
    """
    #Hilbert Transform
    analytic_signal = hilbert(signal)
    #Amplitude Envelope
    amplitude_envelope = np.abs(analytic_signal)
    
    #Temporal Fine Structure
    tfs = analytic_signal.imag/amplitude_envelope
    
    #Convolve amplitude evelope with Gaussian window
    if smooth==True:
        #Gaussian Window
        gauslen = int(fs*glen)
        window = gaussian(gauslen, std=int(gauslen*0.05))
        #Convolve signal for smmothing
        smooth_env = amplitude_envelope.copy()
        smooth_env = sp.convolve(amplitude_envelope,window)
        smooth_env = smooth_env/np.max(smooth_env)
        ini = int(gauslen/2)
        fin = len(smooth_env)-ini
        amplitude_envelope = smooth_env[ini:fin]
    return amplitude_envelope,tfs