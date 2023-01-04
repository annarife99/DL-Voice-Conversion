import sys
import os

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
import plots
import hubconf



vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
#It allows frequency-domain modifications to a digital sound file
#vocoder = hubconf.load_melgan


def OptimStep(model_optim_loss, max_grad):
    for (model, optim, loss, retain_graph) in model_optim_loss:
        model.zero_grad()
        loss.backward(retain_graph=retain_graph)
        clip_grad_norm_(model.parameters(), max_grad)
        optim.step()

def save_checkpoint(model, optimizer, iteration, filepath):
    print("Saving model and optimizer state at iteration {}".format(iteration))
    fp = os.path.join(filepath.split('/')[0],filepath.split('/')[1])
    if os.path.isdir(fp) == False:
        os.mkdir(fp)
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, filepath)


def train_(model, opt, latent_loss_weight, criterion, loader, epochs, inf_iterator_test, logger, iteration):
    c=0
    L_list=[]
    for epoch in range(epochs):
        print('This is epoch number:',epoch)
        mse_sum = 0
        mse_n = 0

        for i, audio in enumerate(loader):
            print('audio shape')
            print(audio.shape)
            print('This is audio number:',i)
            cluster_size = audio.size(1)
            #audio = audio
            
            out, out_conversion, enc_content, spk, latent_loss, idx = model(audio)
            #out: after decoder. reconstruction
            #out_conversion:
            #enc_content: content after codebook
            #spk: speaker info (Sx)
            #latent loss: difference between encoder output and codebook output
            print('ARA')
            print(out.shape)
            print(audio.shape)


            out2=torch.permute(out, (0, 2, 1))
            out=out2



            recon_loss = criterion(out, audio)
            latent_loss = latent_loss.mean()
            L=(recon_loss + latent_loss_weight*latent_loss).data.detach().numpy()
            L_list.append(L)
            print('Loss value:',L)

            OptimStep([(model, opt,  recon_loss + latent_loss_weight*latent_loss , False)], 3)# True),

            mse_sum += recon_loss.item() * audio.shape[0]
            mse_n += audio.shape[0]
            if i% 5 == 0 :
                logger.log_training(iteration = iteration,  loss_recon = recon_loss, latent_loss = latent_loss)

            if i % 200 == 0 :
                model.eval()
                #TAKING THE TEST DATA
                audio = next(inf_iterator_test)
                #audio = audio.cuda()
                out, out_conversion, enc_content, spk, latent_loss, idx = model(audio)

                out2 = torch.permute(out, (0, 2, 1))
                out_conversion2 = torch.permute(out_conversion, (0, 2, 1))
                out = out2
                out_conversion=out_conversion2

                a = torch.stack([audio[0], audio[idx[0]], out[0], out_conversion[0]], dim = 0)
                b = a.detach().cpu().numpy()
                np.save('new_spectograms/audio'+str(c),b)
                a = vocoder.inverse(a) #from frequency domain to sound file
                a = a.detach().cpu().numpy()
                np.save('new_audios/audio'+str(c),a)

                logger.log_validation(iteration = iteration,
                    mel_ori = ("image", plots.plot_spectrogram_to_numpy(), audio[0]),
                    mel_target = ("image",  plots.plot_spectrogram_to_numpy(), audio[idx[0]]),
                    mel_recon = ("image",  plots.plot_spectrogram_to_numpy(), out[0]),
                    mel_conversion = ("image",  plots.plot_spectrogram_to_numpy(), out_conversion[0]),

                    audio_ori = ("audio",48000, a[0]),
                    audio_target = ("audio", 48000, a[1]),
                    audio_recon = ("audio", 48000, a[2]),
                    audio_conversion = ("audio", 48000, a[3]),

                )
                logger.close()
                c+=1


                save_checkpoint(model, opt, iteration, f'checkpoint/model/gen')
                model.train()
            iteration += 1


    plt.plot(L_list)
    plt.xlabel('Iterations')
    plt.ylabel('Loss value')
    plt.savefig('Loss_function.png')
