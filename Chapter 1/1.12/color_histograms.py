# Color Histogram

from matplotlib import pyplot as plt
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)

# grab the image channels
chans = cv2.split(image)
colors = ('b', 'g', 'r')
plt.figure()
plt.title('flattened color histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')

# loop over the image channels
for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)
    plt.xlim([0, 256])

plt.show()



# 2D histograms
# bin: 256 -> 32
fig = plt.figure()

# plot 2D color histogram for green and blue
ax = fig.add_subplot(131)
hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation='nearest')
ax.set_title('2D Color Histogram for G and B')
plt.colorbar(p)

# plot 2D color histogram for green and red
ax = fig.add_subplot(132)
hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation='nearest')
ax.set_title('2D Color Histogram for G and R')
plt.colorbar(p)

# plot 2D color histogram for blue and red
ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation='nearest')
ax.set_title('2D Color Histogram for B and R')
plt.colorbar(p)

plt.show()
# examine the dimension of the 2D histogram
print('2D histogram shape: {}, with {} value'.format(hist.shape, hist.flatten().shape[0]))


# 3D histogram
hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

# display the original image with matplotlib
plt.figure()
plt.axis('off')

plt.imshow(imutils.opencv2matplotlib(image))

plt.show()