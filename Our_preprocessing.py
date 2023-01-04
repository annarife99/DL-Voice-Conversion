import torch
from torch import nn, optim
from torch.nn import functional as F
import sys
import os
import argparse
import librosa
import librosa.display
from librosa.filters import mel as librosa_mel_fn
import numpy as np
import glob
import random
import re
import matplotlib.pyplot as plt

from multiprocessing import Pool
import pickle
#from sklearn.preprocessing import StandardScaler

class Audio2Mel(nn.Module):
    def __init__(self,
        n_fft=1024,#1024
        hop_length=256,
        win_length=1024,
        sampling_rate=24000, #22050,
        n_mel_channels=80, #80
        mel_fmin=0.0,
        mel_fmax=None,
        ):
        super().__init__()
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length).float()
        #It produces a linear transformation matrix to project FFT bins onto Mel-frequency bins
        mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels,  fmin=mel_fmin,  fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        print(fft.shape) #N is the number of frequencies where STFT is applied and T is the total number of frames used.

        real_part, imag_part = fft.unbind(-1)#the last dimension represent the real and imaginary components
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2) #magnitude of the spectogram 
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec


def select_subjects(main_path,number_subjects):
    files=os.path.join(str(main_path),"p*")
    files = sorted(glob.glob(files))
    #Select number of subjects:
    return(random.sample(files, number_subjects))

def select_sentence(select_subj,batch_size):
    str_sentence= os.path.join(str(select_subj),"p*mic1.flac")
    files_sentence = sorted(glob.glob(str_sentence))
    return(random.sample(files_sentence, batch_size))

def load_audio(select_s):
    extract_func = Audio2Mel()
   
    sampling_rate = 24000#22050
    sr= sampling_rate
    x,_ = librosa.load(select_s, sr=sr)
    x, index = librosa.effects.trim(x, top_db=20) #To trim leading and trailing silence
    x=torch.from_numpy(x) #1D audio to a tensor 1x1xn
    x = x[None, None]

    mel = extract_func(x)
    mel = mel.numpy()
    mel = mel[0]
    return mel.astype(np.float32)

def visualize_spect(spec_x):
    d_db = librosa.power_to_db(spec_x,ref=np.max)
    plt.figure(1)
    librosa.display.specshow(d_db, sr=22050,y_axis='mel')
    plt.colorbar()
    plt.title('Spectogram')
    plt.xlabel('Time(s)')
    plt.ylabel('Frequency(Hz)')
    plt.show()


def select_data(main_path,out_dir,number_subjects,batch_size,our_audios):

    if our_audios==True:
        main_path= main_path+'own_audios/'
        s=os.path.join(str(main_path),"p*")
        files = sorted(glob.glob(s))
        for ind,f in enumerate(files):
                m=load_audio(f)
                out_fp = os.path.join(out_dir, f'p{ind}_{ind}.npy')
                np.save(out_fp, m, allow_pickle=False)

    else:
        select_subj= select_subjects(main_path,number_subjects)
        print(select_subj)
        print('Number of subjects',len(np.unique(select_subj)))

        for s in select_subj:
            sub= re.search('p+\d+',s).group(0)
            select_s= select_sentence(s,batch_size)
            print(select_s)
            #x=[load_audio(f) for f in select_s]
            for ind,f in enumerate(select_s):
                m=load_audio(f)
                #visualize_spect(m)
                out_fp = os.path.join(out_dir, f'{sub}_{ind}.npy')
                np.save(out_fp, m, allow_pickle=False)

        #DICTIONARY TO STORE THE DATA: key is the subject and values is a list of the load audio
        #dict_sub[sub]=x
    

if __name__== "__main__":

    out_dir= '/Users/annarife/Desktop/Our_code/output_train'

    main_path= '/Users/annarife/Desktop/DS_10283_3443/VCTK-Corpus-0.92/wav48_silence_trimmed/'
    number_subjects= 109 #Number of subjects chosen randomly
    batch_size= 50 #Number of sentence chosen randomly

    select_data(main_path,out_dir,number_subjects,batch_size,our_audios=False)

    