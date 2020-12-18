# histogram: the frequency distribution of the data
# grayscale and color histograms
# cv2.calcHist(images, channels, mask, histSize, ranges)

# Grayscale Histograms
from matplotlib import pyplot as plt
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# construct a grayscale histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

plt.figure()
plt.axis('off')
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

# plot the histogram
plt.figure()
plt.title('Grayscale histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(hist)
plt.xlim([0, 256])

# normalize the histogram
hist /= hist.sum()

# plot the normalized histogram
plt.figure()
plt.title('Grayscale histogram (Normalized)')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(hist)
plt.xlim([0, 256])
plt.show()

