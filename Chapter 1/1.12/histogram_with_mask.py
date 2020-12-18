from matplotlib import pyplot as plt
import numpy as np
import cv2

def plot_histogram(image, title, mask=None):
    chans = cv2.split(image)
    colors = ('b', 'g', 'r')

    plt.figure()
    plt.title(title)
    plt.xlabel('Bins')
    plt.ylabel('# of label')

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

# load image and plot histogram
image = cv2.imread('beach.png')
cv2.imshow('Original', image)
plot_histogram(image, 'Histogram of the original image')

mask = np.zeros(image.shape[:2], dtype='uint8')
cv2.rectangle(mask, (60, 290), (210, 390), 255, -1)
cv2.imshow('Mask', mask)

masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow('Applying the mask', masked)

# apply plot_histogram(image, title, mask=None)
plot_histogram(image, 'Histogram for masked image', mask=mask)

plt.show()