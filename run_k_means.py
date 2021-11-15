import numpy as np
from numpy.core.numeric import _convolve_dispatcher
from skimage import io
import matplotlib.pyplot as plt


train_image = np.array(io.imread("lena.jpg"), dtype=np.float32) / 255
# plt.imshow(train_image) #Show the train image for sanity check 
# plt.show()

width = train_image.shape[0]
height = train_image.shape[1]
channels = train_image.shape[2]
train_image = train_image.reshape(-1, channels)


# predictions= []

# for i in range(channels):
#     kmeans = KMeans(n_clusters=64, init= 'random', n_init= 10, max_iter= 10, random_state= 0).fit(train_image[:,i].reshape(-1,1))
#     pred= kmeans.predict(train_image[:,i].reshape(-1,1))
#     predictions.append(pred.reshape(width, height))


# prediction = np.zeros((width, height, 3))*255

# """Add the channels to the needed image one by one"""
# prediction[:,:,0] = predictions[0]
# prediction[:,:,1] = predictions[1]
# prediction[:,:,2] = predictions[2]

# kmeans from scratch=============================================
max_iters = 8
k = 16

index = np.zeros(
    (train_image.shape[0],), dtype=int
)  # dtype= int, because it will be used later as index for another array

"""Iterate through each pixel and compare the values with the centroid value
Initializing the centroids by taking random pixel values from the imaage (First shuffle it and then take the first k pixels)"""

rand = np.random.randint(low=0, high= train_image.shape[0],size=k)
# print(rand)
centroids = train_image[rand, :]  # k x n dimensional matrix
# print(f'centroids 1 iter: {centroids}') #Print the centroids after initialization for sanity check

diff = np.zeros((k,))
for iter in range(max_iters):
    print(f"Iteration: {iter+1}==============================================")
    for i in range(train_image.shape[0]):
        for color in range(k):
            diff[color] = np.sum((train_image[i, :] - centroids[color, :]) ** 2)
        index[i] = np.argmin(diff)
    centroids = np.zeros((k, train_image.shape[1]))
    
    for color in range(k):
        if np.sum(index==color)!=0:
            Mean = (train_image[index == color, :]).mean(axis=0)
            centroids[color] = Mean
    # print(f'centroids {iter+2} iter: {centroids}')    #Print the centroids after the updates for sanity check
prediction = centroids[index, :]
prediction = prediction.reshape(width, height, channels)
plt.imshow(prediction)
plt.show()