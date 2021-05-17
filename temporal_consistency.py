import os, math, glob
from pathlib import Path
import numpy as np
from imageio import imread, imsave
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
# import pytorch_lightning as pl
# from pytorch_lightning import Trainer
# from multiprocessing import Process

from models.seq2seq_ConvLSTM import EncoderDecoderConvLSTM

# from torchsummary import summary
from easydict import EasyDict as edict


from tqdm import tqdm as tqdm
from utils.meter import AverageValueMeter
from utils.meter import accuracy

from evaluate import conv_lstm_inference

import wandb
import hydra

# TV-loss
import kornia
tv_loss = kornia.losses.TotalVariation()
tv_loss.__name__ = 'tv_loss'


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
def soft_dice_loss(input:torch.Tensor, target:torch.Tensor, eps=1):
    input_sigmoid = torch.sigmoid(input)
    # eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + eps) /
                (iflat.sum() + tflat.sum() + eps))

# refer to https://blog.medisolv.com/articles/evaluating-the-power-of-predictive-analytics-statistics-basics-for-clinicians-and-quality-professionals
def soft_specificity_loss(input:torch.Tensor, target:torch.Tensor, eps=1):
    input_sigmoid = torch.sigmoid(input)
    # eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    
    total_neg = (1 - tflat).sum()
    true_neg = ((1 - tflat) * (1 - iflat)).sum()

    return 1 - ((true_neg + eps) / (total_neg + eps))

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

    else:
        
        _CWD_ = Path(hydra.utils.get_original_cwd()) 
        # DATA = np.load(Path(_CWD_ / "data" / "elephant_patchsize_16.npy")
    
        npy_url = glob.glob(str(_CWD_ / f"data/{cfg.data.name}_{cfg.data.folder}*{cfg.data.note}.npy"))[0]
        

        print(f"npyname: {os.path.split(npy_url)[-1]}")
        DATA = np.load(npy_url)

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

    _CWD_ = Path(hydra.utils.get_original_cwd()) 

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataloader = get_dataloader(cfg)
    model = EncoderDecoderConvLSTM(nf=cfg.model.number_filters, in_chan=cfg.model.input_channels)
    model.to(DEVICE)

    optimizer = optim.SGD(model.parameters(), 
                            lr=cfg.model.learning_rate, 
                            momentum=cfg.model.momentum)

    per_epoch_steps = 7000 // cfg.model.batch_size
    total_training_steps = cfg.model.max_epoch * per_epoch_steps
    warmup_steps = cfg.model.warmup_coef * per_epoch_steps
    if cfg.model.use_lr_scheduler:
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_training_steps)
    
    soft_dice_loss.__name__ = 'dice_loss'
    soft_specificity_loss.__name__ = 'spec_loss'
    temporal_consistency_loss.__name__ = 'tc_loss'
    metrics = [soft_dice_loss, temporal_consistency_loss, soft_specificity_loss]

    tcloss_valid_his = []
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
                # dice_loss = soft_dice_loss(output, y)
                dice_loss = soft_dice_loss(output, y)
                # tv_loss_ = 1e-5 * torch.mean(tv_loss(y_pred))

                total_loss =  (1-cfg.model.tc) * dice_loss + cfg.model.tc * tc_loss
                # print(f"epoch: {epoch}, loss: {total_loss}, tc_loss: {tc_loss}")

                # print(loss.shape)

                if 'train' == phase:
                    total_loss.backward()
                    optimizer.step()

                    if cfg.model.use_lr_scheduler:
                        lr_scheduler.step()

                total_loss_value = total_loss.cpu().detach().numpy()
                loss_meter.add(total_loss_value)

                # update metrics logs
                for metric_fn in metrics:
                    metric_value = metric_fn(output, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}

            # print(f"lr: {lr_scheduler.get_lr()}")
            currlr = lr_scheduler.get_last_lr()[0] if cfg.model.use_lr_scheduler else cfg.model.learning_rate
            print(f"epoch ({phase}): {epoch+1}/{cfg.model.max_epoch}, loss: {loss_meter.mean}, dice_loss: {metrics_logs['dice_loss']}, tc_loss: {metrics_logs['tc_loss']}, lr: {currlr}")
            wandb.log({phase: {\
                'total_loss': loss_meter.mean, \
                'dice_loss': metrics_logs['dice_loss'],\
                'spec_loss': metrics_logs['spec_loss'],\
                'tc_loss': metrics_logs['tc_loss']}, \
                'lr': currlr, \
                'epoch': epoch+1})

            if 'valid'==phase: 
                tcloss_valid_his += [metrics_logs['tc_loss']]

        
        tcloss_valid_mean = sum(tcloss_valid_his[:-1]) / (len(tcloss_valid_his[:-1]) + 1e-6)
        anamoly = abs(tcloss_valid_his[-1] - tcloss_valid_mean)
        

        anamolyFlag = False
        if (epoch>10) and (anamoly>0.15):
            anamolyFlag = True
            print(f"epoch: {epoch}, anamolyFlag: {anamolyFlag}")
            tcloss_valid_his = tcloss_valid_his[:-1]

        wandb.log({"anamoly": anamoly, "anamolyFlag": 1 if anamolyFlag else 0, 'epoch': epoch+1})
        
        if len(tcloss_valid_his) > 5: tcloss_valid_his = tcloss_valid_his[-5:]

        if (epoch < 5) or ((epoch+1) % cfg.model.logImgPerEp == 0) or anamolyFlag:
            # model inference
            data_folder = _CWD_ / "data" / f"{cfg.data.name}" / f"{cfg.data.folder}"

            # data_folder = Path(f"/home/omegazhangpzh/temporal-consistency/data/elephant_hill/{cfg.data.sat}_data")
            masks = conv_lstm_inference(model, data_folder, patchsize=cfg.model.inferPatchSize).squeeze()

            predMaskDir = Path(cfg.experiment.output) / f"predMasks_ep{epoch+1}"
            maskArrDir = Path(cfg.experiment.output) / "maskArr"
            print(f"maskArrDir: {predMaskDir}")

            for saveDir in [predMaskDir, maskArrDir]:
                if not os.path.exists(saveDir): os.makedirs(saveDir)
        
            # mask_list = [masks[idx,] for idx in range(0, masks.shape[0])]
            mask_list = []
            for idx in range(0, masks.shape[0]):
                mask_list += [masks[idx,]]

                if cfg.model.saveImgSglFlag:
                    imsave(predMaskDir / f"frame_{idx}.png", np.uint8(masks[idx,]*255))


            maskArr = np.concatenate(tuple(mask_list), axis=1)
            wandb.log({f"predMasks/{cfg.data.name}_ep{epoch+1}": wandb.Image(maskArr)})
            # wandb.log({f"predMasks/{cfg.data.name}": plt.imshow(maskArr, cmap='hsv', vmin=1, vmax=1)})

            if cfg.model.saveImgArrFlag:
                imsave(maskArrDir / f"{cfg.data.name}_ep{epoch+1}.png", np.uint8(maskArr*255))






