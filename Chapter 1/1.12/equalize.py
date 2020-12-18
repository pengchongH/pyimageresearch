# Histogram equalization is applied to grayscale images.
# improves the contrast of an image by “stretching” the distribution of pixels

import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply an equalization to image
eq = cv2.equalizeHist(gray)

cv2.imshow('Original(gray)', gray)
cv2.imshow('Equalized', eq)
cv2.waitKey(0)
