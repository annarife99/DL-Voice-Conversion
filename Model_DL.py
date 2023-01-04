import os
import random
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
import vq_model2
import vq_model3
import vq_model
import vq_model_simple
import train_normal


class AudioNpyLoader():
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.audios = os.listdir(self.audio_path)
        random.seed(1234)
        random.shuffle(self.audios)
    def __getitem__(self, index):
        item = f'{self.audio_path}/{self.audios[index]}'
        item = np.load(item, allow_pickle=True)
        return item
    def __len__(self):
        return len(self.audios)

class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)
    def log_training(self, iteration,figure = "training.loss", **kwarg):
        self.add_scalars(figure, kwarg , iteration)
    def log_validation(self, iteration, **kwarg):
        for key in (kwarg.keys()):
            (type_, method_, data) = kwarg[key]
            if type_=="audio":
                self.add_audio(
                f'{key}',
                data, iteration, sample_rate=method_)
            elif type_ == "scalars":
                self.add_scalars("validation.loss", data, iteration)
            elif type_ == "image":
                data = data.detach().cpu().numpy()
                self.add_image(
                f'{key}',
                method_(data),
                iteration, dataformats='HWC')


def VCTK_collate(batch):
    maxn = 128
    audio = []
    #name = []
    for item in batch:
        item_len = int(item.shape[1])
        
        if item_len>maxn:
            rand = np.random.randint(item_len-maxn)
            item_128 = item[:,rand:rand+maxn]
        else:
            item_128 = item
        audio += [item_128]
    for i in range(len(audio)):
        a = audio[i]
        a = np.pad(a,((0,0),(0,maxn-a.shape[1])),'reflect')
        audio[i] = a
    return torch.tensor((np.array(audio)))

def make_inf_iterator(data_iterator):
    while True:
        for data in data_iterator:
            yield data

def prepare_directories_and_logger(logger_class, output_directory = 'output', 
	log_directory = 'log'):
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    logger = logger_class(os.path.join(output_directory, log_directory))
    return logger


if __name__== "__main__":
   
    logger = prepare_directories_and_logger(Logger, output_directory = 'results/modelDL')

    #Train Dataset
    audio_path= '/Users/annarife/Desktop/Our_code/train/'
    dataTrain= AudioNpyLoader(audio_path)
    loader = DataLoader(dataTrain, batch_size=1, shuffle=True,collate_fn=VCTK_collate)

    #Test Dataset
    audio_path_test= '/Users/annarife/Desktop/Our_code/test/'
    dataTest = AudioNpyLoader(audio_path_test)
    test_loader = DataLoader(dataTest, batch_size=1, shuffle=True,collate_fn=VCTK_collate)
    inf_iterator_test = make_inf_iterator(test_loader)

    n= 128#128 #number of vectors in the codebook
    ch= 128 #channels in encoder and decoder

    #model initialization,  B x F x T tensor , where F is the number of input channels
    model = vq_model2.VC_MODEL(in_channel=80,channel=ch,n_embed=n)
    #model=vq_model_simple.VC_MODEL(in_channel=80,channel=ch,n_embed=n)
    opt = optim.Adam(model.parameters())
    for el in model.parameters():
        print(el.shape)

    #Training
    criterion = nn.L1Loss()
    latent_loss_weight = 0.1
    iteration = 0


    #Run model
    train_normal.train_(model, opt, latent_loss_weight, criterion, loader,3, inf_iterator_test, logger, iteration)
    print('FINISH')
    


    


    
   
    
 

    