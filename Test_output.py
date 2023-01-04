import librosa
import librosa.display
from librosa.filters import mel as librosa_mel_fn
import soundfile as sf
import numpy as np
import glob
import random
import re
import matplotlib.pyplot as plt
import os


def visualize_timeSeries(i,np_file):
    x=np.load(np_file)
    plt.figure(i)
    fig, axs = plt.subplots(2, 2,figsize=(12,6))
    (ax1, ax2), (ax3, ax4) = axs
    ax1.plot(x[0,:])
    ax1.title.set_text('Original Audio')
    ax2.plot(x[1,:])
    ax2.title.set_text('Target Audio')
    ax3.plot(x[2,:])
    ax3.title.set_text('Reconstructed Audio')
    ax4.plot(x[3,:])
    ax4.title.set_text('Converted Audio')
    plt.savefig('/Users/annarife/Desktop/Our_code/plots/test10/time_serie'+str(i))
    
def visualize_spectograms(i,np_file):
    x=np.load(np_file)
    plt.figure(i)
    fig, axs = plt.subplots(2, 2,figsize=(12,6))
    (ax1, ax2), (ax3, ax4) = axs
    im1=ax1.imshow(x[0,:], aspect="auto", origin="lower",interpolation='none')
    ax1.title.set_text('Original Audio')
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("Channels")
    plt.tight_layout()
    plt.colorbar(im1, ax=ax1)

    im2=ax2.imshow(x[1,:], aspect="auto", origin="lower",interpolation='none')
    ax2.title.set_text('Target Audio')
    ax2.set_xlabel("Frames")
    ax2.set_ylabel("Channels")
    plt.tight_layout()
    plt.colorbar(im2, ax=ax2)

    im3=ax3.imshow(x[2,:], aspect="auto", origin="lower",interpolation='none')
    ax3.title.set_text('Reconstructed Audio')
    ax3.set_xlabel("Frames")
    ax3.set_ylabel("Channels")
    plt.tight_layout()
    plt.colorbar(im3, ax=ax3)

    im4=ax4.imshow(x[3,:], aspect="auto", origin="lower",interpolation='none')
    ax4.title.set_text('Converted Audio')
    ax4.set_xlabel("Frames")
    ax4.set_ylabel("Channels")
    plt.tight_layout()
    plt.colorbar(im4, ax=ax4)
    plt.savefig('/Users/annarife/Desktop/Our_code/plots/test10/specto'+str(i))

def convert_file_audio(i,file):
    x=np.load(file)
    original_audio=x[0,:]
    targ_audio=x[1,:]

    rec_audio=x[2,:]
    conv_audio=x[3,:]
    sr=24000

    sf.write('plots/test10/audios/original_audio'+str(i)+'.wav', original_audio, sr, subtype='PCM_24')
    sf.write('plots/test10/audios/targ_audio'+str(i)+'.wav', targ_audio, sr, subtype='PCM_24')
    sf.write('plots/test10/audios/rec_audio'+str(i)+'.wav', rec_audio, sr, subtype='PCM_24')
    sf.write('plots/test10/audios/conv_audio'+str(i)+'.wav', conv_audio, sr, subtype='PCM_24')
   
    


    
if __name__== "__main__":

    out_dir_t= '/Users/annarife/Desktop/new_audios/'
    out_dir_s= '/Users/annarife/Desktop/new_spectograms/'

    files_t=os.path.join(str(out_dir_t),"audio*")
    files_t = sorted(glob.glob(files_t))

    files_s=os.path.join(str(out_dir_s),"audio*")
    files_s = sorted(glob.glob(files_s))

    for i,audio in enumerate(files_t):
        visualize_timeSeries(i,audio)
        convert_file_audio(i,audio)

    for i,audio in enumerate(files_s):
        visualize_spectograms(i,audio)
   
