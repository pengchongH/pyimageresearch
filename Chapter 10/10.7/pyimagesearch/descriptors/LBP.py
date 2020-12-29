from skimage import feature
import numpy as np

class LocalBinaryPatterns:
    def __init__(self, numpoints, radius):
        self.numpoints =numpoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute LBP represntation of the image and build histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numpoints, self.radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=range(0, self.numpoints + 3), range=(0, self.numpoints + 2))

        # normalize the histogram
        hist = hist.astype('float')
        hist /= (hist.sum() + eps)

        return hist

