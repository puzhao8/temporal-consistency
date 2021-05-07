import os, math, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from multiprocessing import Process

from models.seq2seq_ConvLSTM import EncoderDecoderConvLSTM

# from torchsummary import summary
from easydict import EasyDict as edict


from tqdm import tqdm as tqdm
from utils.meter import AverageValueMeter
from utils.meter import accuracy

from evaluate import conv_lstm_inference

import wandb
import hydra

from torch.optim.lr_scheduler import LambdaLR
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    """ <Borrowed from `transformers`>
        Create a schedule with a learning rate that decreases from the initial lr set in the optimizer to 0,
        after a warmup period during which it increases from 0 to the initial lr set in the optimizer.
        Args:
            optimizer (:class:`~torch.optim.Optimizer`): The optimizer for which to schedule the learning rate.
            num_warmup_steps (:obj:`int`): The number of steps for the warmup phase.
            num_training_steps (:obj:`int`): The total number of training steps.
            last_epoch (:obj:`int`, `optional`, defaults to -1): The index of the last epoch when resuming training.
        Return:
            :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))  # this is correct
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

# borrowed from: https://github.com/SebastianHafner/urban_dl/blob/master/experiment_manager/loss.py  
def soft_dice_loss(input:torch.Tensor, target:torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection) /
                (iflat.sum() + tflat.sum() + eps))

def temporal_consistency_loss(input, y):
    output = torch.sigmoid(input)
    temporal_consistency_loss = 0
    for t in range(y.shape[1]-1):
        temporal_consistency_loss += nn.MSELoss()(output[:,t,...], y[:,t,...])
    
    return temporal_consistency_loss / (y.shape[1]-1) 


def get_dataloader(cfg):
    
    if cfg.data.name == 'rand':
        # b, t, c, h, w
        x = torch.randn(100, 5, 3, 11, 11)
        y = torch.ones(100, 5, 1, 11, 11)
        trainset = torch.utils.data.TensorDataset(torch.Tensor(x), torch.Tensor(y))
        print(f"x: [{x.min()}, {x.max()}]")

        dataloader = edict()
        dataloader['train'] = torch.utils.data.DataLoader(trainset, batch_size=cfg.model.batch_size, shuffle=True)
        dataloader['valid'] = torch.utils.data.DataLoader(trainset, batch_size=cfg.model.batch_size, shuffle=True)

        return dataloader

    if cfg.data.name == 'elephant':
        # DATA = np.load(Path(hydra.utils.get_original_cwd()) / "data" / "elephant_patchsize_16.npy")
        if cfg.data.sat == 's2':
            npyname = glob.glob("/home/omegazhangpzh/temporal-consistency/data/elephant_s2_*.npy")[0]
        
        if cfg.data.sat == 's1':
            npyname = glob.glob("/home/omegazhangpzh/temporal-consistency/data/elephant_s1_*.npy")[0]

        DATA = np.load(npyname)

        inputs = DATA[:, :, :3, ...]
        labels = DATA[:, :, 3:4, ...]
        labels_ = DATA[:, :, -1:, ...]

        from sklearn.model_selection import train_test_split
        train_X, valid_X, train_y, valid_y = train_test_split(inputs, labels, train_size=cfg.data.train_size, random_state=42)   
        # train_X, valid_X, train_y, valid_y = train_test_split(inputs[:1000], labels[:1000], train_size=cfg.data.train_size, random_state=42)   

        print(train_X.shape, train_y.shape)

        trainset = torch.utils.data.TensorDataset(torch.Tensor(train_X), torch.Tensor(train_y))
        validset = torch.utils.data.TensorDataset(torch.Tensor(valid_X), torch.Tensor(valid_y))
    
        dataloader = edict()
        dataloader['train'] = torch.utils.data.DataLoader(trainset, batch_size=cfg.model.batch_size, shuffle=True)
        dataloader['valid'] = torch.utils.data.DataLoader(validset, batch_size=cfg.model.batch_size, shuffle=True)

        return dataloader



def train_conv_lstm(cfg):
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    
    dataloader = get_dataloader(cfg)
    model = EncoderDecoderConvLSTM(nf=cfg.model.number_filters, in_chan=cfg.model.input_channels)
    model.to(DEVICE)

    optimizer = optim.SGD(model.parameters(), 
                            lr=cfg.model.learning_rate, 
                            momentum=cfg.model.momentum)

    per_epoch_steps = 7000 // cfg.model.batch_size
    total_training_steps = cfg.model.max_epoch * per_epoch_steps
    warmup_steps = 5 * per_epoch_steps
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_training_steps)
    
    soft_dice_loss.__name__ = 'dice_loss'
    temporal_consistency_loss.__name__ = 'tc_loss'
    metrics = [soft_dice_loss, temporal_consistency_loss]

    for epoch in range(cfg.model.max_epoch):
        for phase in ['train', 'valid']:

            if 'train' == phase: model.train()
            if 'valid' == phase: model.eval()

            loss_meter = AverageValueMeter()
            metrics_meters = {metric.__name__: AverageValueMeter() for metric in metrics}

            for (x, y) in dataloader[phase]:
                """ batch update """
            
                # get the inputs; data is a list of [inputs, labels]
                x, y = x.to(DEVICE), y.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = model.forward(x, future_seq=cfg.model.future_seq)
                
                tc_loss = temporal_consistency_loss(output, y)
                dice_loss = soft_dice_loss(output, y)

                total_loss =  dice_loss + cfg.model.tc * tc_loss

                # print(f"epoch: {epoch}, loss: {total_loss}, tc_loss: {tc_loss}")

                # print(loss.shape)

                if 'train' == phase:
                    total_loss.backward()
                    optimizer.step()
                    lr_scheduler.step()

                total_loss_value = total_loss.cpu().detach().numpy()
                loss_meter.add(total_loss_value)

                # update metrics logs
                for metric_fn in metrics:
                    metric_value = metric_fn(output, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}

            # print(f"lr: {lr_scheduler.get_lr()}")
            print(f"epoch ({phase}): {epoch}/{cfg.model.max_epoch}, loss: {loss_meter.mean}, dice_loss: {metrics_logs['dice_loss']}, tc_loss: {metrics_logs['tc_loss']}, lr: {lr_scheduler.get_lr()[0]}")
            wandb.log({phase: {\
                'total_loss': loss_meter.mean, \
                'dice_loss': metrics_logs['dice_loss'],\
                'tc_loss': metrics_logs['tc_loss']}, \
                'lr': lr_scheduler.get_lr()[0], \
                'epoch': epoch+1})


    
        if epoch % 10 == 0:
            # model inference
            # data_folder = Path(hydra.utils.get_original_cwd()) / "data" / "elephant_hill" / "sentinel2_data")
            data_folder = Path("/home/omegazhangpzh/temporal-consistency/data/elephant_hill/sentinel2_data")
            masks = conv_lstm_inference(model, data_folder).squeeze()

            mask_list = [masks[idx,] for idx in range(0, masks.shape[0])]
            maskArr = np.concatenate(tuple(mask_list), axis=1)
            wandb.log({f"predMasks/{cfg.data.name}_ep{epoch}": wandb.Image(maskArr)})
            # wandb.log({f"predMasks/{cfg.data.name}": plt.imshow(maskArr, cmap='hsv', vmin=1, vmax=1)})

        






