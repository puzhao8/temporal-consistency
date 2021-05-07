
# def conv_lstm_inference(model, folder):
# model.cpu()
patchsize = 512

input_patchsize = 2 * patchsize
padSize = int(patchsize/2)

# read data
data_dir = "path_to_input_image_series"
images = tuple()
for filename in sorted(os.listdir(data_dir)):
    image = imread(data_dir / filename) / 255.0
    images = images + (image,)

T, H, W, C = img.shape
img_pad0 = zero_padding(img, patchsize) # pad img into a shape: (m*PATCHSIZE, n*PATCHSIZE)
img_pad = np.pad(img_pad0, ((0, 0) (padSize, padSize), (padSize, padSize), (0, 0)), mode='symmetric')

in_tensor = torch.from_numpy(img_pad.transpose(2, 0, 1)).unsqueeze(0)

# (Height, Width, Channels) = img_pad.shape
# pred_mask_pad = np.zeros((Height, Width))
# for i in tqdm(range(0, Height - input_patchsize + 1, patchsize)):
#     for j in range(0, Width - input_patchsize + 1, patchsize):
#         # print(i, i+input_patchsize, j, j+input_patchsize)
#         inputPatch = in_tensor[..., i:i+input_patchsize, j:j+input_patchsize]

#         predPatch = model.forward(inputPatch.type(torch.cuda.FloatTensor))

#         predPatch = predPatch.squeeze().cpu().detach().numpy()#.round()
#         pred_mask_pad[i+padSize:i+padSize+patchsize, j+padSize:j+padSize+patchsize] = predPatch[padSize:padSize+patchsize, padSize:padSize+patchsize]  # need to modify

# pred_mask = pred_mask_pad[padSize:padSize+H, padSize:padSize+W] # clip back to original shape
