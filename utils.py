import matplotlib.pyplot as plt

def plotimages(train_image, prediction):
    #Plot images side by side (compressed on the right)
    _,ax= plt.subplots(nrows= 1, ncols= 2)
    ax[0].imshow(train_image)
    ax[1].imshow(prediction)
    plt.show()
    
def piechart(values, colors):
    plt.pie(values, colors= colors)
    plt.show()