
#%%%
from imageio import imread, imsave
from sklearn.cluster import KMeans

img = imread("./data\elephant\s2_data\MSI_20170730_S2.png")

# %%
import numpy as np 

X = np.array([[1, 2], [1, 4], [1, 0],
           [10, 2], [10, 4], [10, 0]])
# %%
X.shape
# %%
