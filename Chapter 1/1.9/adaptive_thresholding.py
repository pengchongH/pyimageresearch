# Adaptive Thresholding
# Assumption: smaller regions of an image are more likely to have approximately uniform illumination

from skimage.filters import threshold_local
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
cv2.imshow('Image', image)

# apply adaptive thresholding
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
cv2.imshow('OpenCV Mean Thresh', thresh)

# apply scikit-image adaptive thresholding
T = threshold_local(blurred, 29, offset=5, method='gaussian')
thresh = (blurred < T).astype('uint8') * 255
cv2.imshow('scikit-image Mean Thresh', thresh)
cv2.waitKey(0)
