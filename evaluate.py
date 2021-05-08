from pathlib import Path 
import os
import numpy as np 
from imageio import imread, imsave
import torch
from tqdm import tqdm

def zero_padding(arr, patchsize):
    # print("zero_padding patchsize: {}".format(patchsize))
    (t, h, w, c) = arr.shape
    pad_h = (1 + np.floor(h/patchsize)) * patchsize - h
    pad_w = (1 + np.floor(w/patchsize)) * patchsize - w

    arr_pad = np.pad(arr, ((0, 0), (0, int(pad_h)), (0, int(pad_w)), (0, 0)), mode='symmetric')
    return arr_pad


def arrange_conv_lstm_predMasks(images):
    img_arr = ()
    for k in range(0, images.shape[0]):
        img_arr += (np.concatenate([images[k,i,:,:].squeeze() for i in range(images.shape[1])], axis=1), )
    img_arr = np.concatenate(img_arr, axis=0)

    return img_arr


def conv_lstm_inference(model, datafolder, patchsize=128):
    # model.cpu()
    # patchsize = 128

    input_patchsize = 2 * patchsize
    padSize = int(patchsize/2)

    # read data
    images = tuple()
    for filename in sorted(os.listdir(datafolder)):
        image = imread(datafolder / filename) / 255.0
        images = images + (image,)
    img_ts_arr = np.stack(images, axis=0)

    print(img_ts_arr.shape)
    T, H, W, C = img_ts_arr.shape
    img_pad0 = zero_padding(img_ts_arr, patchsize) # pad img into a shape: (m*PATCHSIZE, n*PATCHSIZE)
    # print(img_pad0)

    img_pad1 = np.pad(img_pad0, ((0, 0), (padSize, padSize), (padSize, padSize), (0, 0)), mode='symmetric')
    in_tensor = torch.from_numpy(img_pad1.transpose(0, 3, 1, 2)).unsqueeze(0)

    (B, T, Channels, Height, Width) = in_tensor.shape
    # seq_len = 5
    # pred_mask_tuples = ()
    # for t_start in range(0, T-seq_len):
    #     img_pad = in_tensor[:, t_start:t_start+5,]

    pred_mask_pad = np.zeros((T, Height, Width))
    for i in tqdm(range(0, Height - input_patchsize + 1, patchsize)):
        for j in range(0, Width - input_patchsize + 1, patchsize):
            # print(i, i+input_patchsize, j, j+input_patchsize)
            inputPatch = in_tensor[..., i:i+input_patchsize, j:j+input_patchsize]
            # print(inputPatch.shape)
            predPatch = model.forward(inputPatch.type(torch.cuda.FloatTensor)) # t x 1 x H x W
            predPatch = torch.sigmoid(predPatch).squeeze().cpu().detach().numpy()#.round() # t x H x W
            pred_mask_pad[:, i+padSize:i+padSize+patchsize, j+padSize:j+padSize+patchsize] = predPatch[:, padSize:padSize+patchsize, padSize:padSize+patchsize]  # need to modify

    pred_mask = pred_mask_pad[:, padSize:padSize+H, padSize:padSize+W] # clip back to original shape
    # pred_mask_tuples += (pred_mask,)

    # masks = np.stack(pred_mask_tuples, axis=0)
    # full_prediction = arrange_conv_lstm_predMasks(masks)

    return pred_mask  



if __name__ == "__main__":
    # read data
    data_dir = Path("/home/omegazhangpzh/temporal-consistency/data/elephant_hill/sentinel2_data")
    
    # model = ...
    masks = conv_lstm_inference(model, folder)