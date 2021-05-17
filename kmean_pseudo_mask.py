
from imageio import imread, imsave
import os, cv2 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
import statistics
from pathlib import Path

def get_ref_hr(mask_pred, ref_lr):
    mask = mask_pred.copy()
    ref_hr = np.zeros(mask.shape)
    for i in np.unique(mask):
        loc = np.where(mask==i)
        print(i, statistics.mode(ref_lr[loc]))
        ref_hr[loc] = statistics.mode(ref_lr[loc])

    return ref_hr

num_classes = 5
train_size = 1e4

workspace = Path("E:\PyProjects/temporal-consistency\data\elephant")
km_dir = workspace / "s1_kmap_km"
km_mask_dir = workspace / "s1_kmap_mask_km"

for folder in [km_dir, km_mask_dir]:
    if not os.path.exists(folder): os.makedirs(folder)

for filename in os.listdir(workspace / "s1_kmap"):
    img = imread(workspace / "s1_kmap" / filename)
    ref_lr_ = imread(workspace / "s1_kmap_mask" / filename)
    ref_lr = cv2.resize(ref_lr_, (img.shape[1], img.shape[0]))

    img_vec = img.reshape(-1, 3) / 255
    X = img_vec[np.random.randint(0, img_vec.shape[0], int(train_size)),]

    kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(X)
    my_cmap = ListedColormap(kmeans.cluster_centers_)

    # kmeans clustering
    y_pred = kmeans.predict(img_vec)
    mask_pred = y_pred.reshape(img.shape[:2])
    print(f"pred unique: {np.unique(y_pred)}")

    plt.imsave(km_dir / f"{filename[:-4]}_clustered.png", mask_pred, cmap=my_cmap)

    ref_hr = get_ref_hr(mask_pred, ref_lr)
    imsave(km_mask_dir / f"{filename}", ref_hr)

    # plt.figure(figsize=(15,15))
    # plt.imshow(ref_hr, cmap='gray')
