# Image compression from scratch using K-means algorithm and also by using K-means from Scikit Learn

* In this project, K-means algorithm (Which is an unsupervised learning algorithm) has been written from scratch and image compression has been done using it. 
The number of clusters chosen for K-means image compression is 16, that means the original image is being compressed to having only 16 RGB colors all over the image. A better and smoother output will be retrieved at larger image file size by increasing the values of K (Try 32,64,128 etc)
First, the input image of shape (height, width, channels) has been converted to (height*width, channels) to get RGB values for each pixel and then K-means has been applied.
Run this file in bash or cmd: `python run_k_means.py` 

* The same thing has also been done using K-means from Scikit Learn package. Bash/ Cmd Command: `python kmeans_sklearn.py`

* In the end, the before and after compression images are plotted along with the color palette of the compressed image in a pie chart (Functions written in utils.py file)

