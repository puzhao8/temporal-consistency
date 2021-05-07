


import os, cv2
from pathlib import Path 
import datetime
import numpy as np 

from imageio import imread, imsave


def get_random_idx_tuples(seq_len=5, patchsize=128, samples_per_class=100):
    (T, _, H, W) = DATA.shape
    num_dates=DATA.shape[0]

    # randomly choose time start 
    t_start = np.random.randint(0, num_dates-seq_len, 1)[0]
    t_end = t_start + seq_len

    # random blanced sampling according to train mask
    mask_t_end = DATA[t_end, -2, ...]

    # burned vs. unburned at time t_end
    burned_coord = np.stack(np.where(mask_t_end==1), axis=0).transpose()
    unburned_coord = np.stack(np.where(mask_t_end==0), axis=0).transpose()

    # randomly generate index
    rand_idx1 = np.random.randint(0, len(burned_coord), samples_per_class)
    rand_idx2 = np.random.randint(0, len(unburned_coord), samples_per_class)

    burnedIdxTuple = [(t_start, t_end, burned_coord[idx,0], burned_coord[idx, 1]) for idx in list(rand_idx1) \
        if (burned_coord[idx,0] in range(patchsize, H-patchsize)) and (burned_coord[idx,1] in range(patchsize, W-patchsize))]
    unburnedIdxTuple = [(t_start, t_end, unburned_coord[idx,0], unburned_coord[idx, 1]) for idx in list(rand_idx2) \
        if (unburned_coord[idx,0] in range(patchsize, H-patchsize)) and (unburned_coord[idx,1] in range(patchsize, W-patchsize))]

    return burnedIdxTuple + unburnedIdxTuple

data_folder = Path("E:\PyProjects/temporal-consistency\data\elephant_hill")

data_name = 'sentinel1'
data_dir = data_folder / f"{data_name}_data"
trainMask_dir = data_folder / f"{data_name}_mask"
validMask_dir = data_folder / f"{data_name}_mask_fusion"


trainMasks = tuple()
validMasks = tuple()
images = tuple()
for filename in sorted(os.listdir(data_dir)):
    
    image = imread(data_dir / filename) / 255.0
    images = images + (image,)

    trainMask = imread(trainMask_dir / filename) / 255.0
    validMask = imread(validMask_dir / filename) / 255.0

    trainMask = cv2.resize(trainMask, (image.shape[1], image.shape[0]))
    validMask = cv2.resize(validMask, (image.shape[1], image.shape[0]))

    trainMasks = trainMasks + (trainMask[:,:,np.newaxis],)
    validMasks = validMasks + (validMask[:,:,np.newaxis],)

    print(filename)
    print(len(images))


image_stack = np.stack(images, axis=0)
train_mask_stack = np.stack(trainMasks, axis=0)
valid_mask_stack = np.stack(validMasks, axis=0)

# T x C x H x W
DATA = np.concatenate((image_stack, train_mask_stack, valid_mask_stack), axis=-1).transpose(0,3,1,2)

patchsize = 16
seq_len = 10

index_tuple_set = []
for i in range(5):
    index_tuple_set += get_random_idx_tuples(seq_len=seq_len, patchsize=patchsize, samples_per_class=1000)
    # print(len(index_tuple_set))
    # print(index_tuple_set)



def get_patches(idx, patchsize=patchsize):
    half_patchsize = patchsize // 2
    return DATA[idx[0]:idx[1], :, \
        (idx[2]-half_patchsize):(idx[2]+half_patchsize),\
        (idx[3]-half_patchsize):(idx[3]+half_patchsize)]

samples = np.stack(tuple(map(get_patches, index_tuple_set)), axis=0)
n, l, c, w, w = samples.shape
np.save(f"data/elephant_{data_name}_{n}x{l}x{c}x{w}x{w}.npy", samples)
# print(half_patchsize)
# train_mask_stack

