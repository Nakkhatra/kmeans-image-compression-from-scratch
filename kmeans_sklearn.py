from sklearn.cluster import KMeans
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import utils

train_image = np.array(io.imread("bird_small.png"), dtype=np.float32) / 255
# plt.imshow(train_image) #Show the train image for sanity check 
# plt.show()

height = train_image.shape[0]
width = train_image.shape[1]
channels = train_image.shape[2]
train_image_ = train_image.reshape(-1, channels)
n_clusters= 64

kmeans = KMeans(n_clusters= n_clusters, init= 'random', n_init= 10, max_iter= 10, random_state= 0).fit(train_image_)

index= kmeans.labels_   #This gives the indices which is of shape train_image.shape[0]xn ((height*widht), channels)
centroids= kmeans.cluster_centers_  #This gives the centroids which is of shape kxn (k, channels)
prediction = centroids[index,:].reshape(height, width, channels)

print(type(centroids))

#Plot train image and compressed image side by side
utils.plotimages(train_image, prediction)

#Plot color palette
utils.piechart(values= np.ones((n_clusters,)), colors= centroids)



