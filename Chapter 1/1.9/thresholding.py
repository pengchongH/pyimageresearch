# Thresholding
# the binarization of an image
# USE: focus on objects or areas of particular interest in an image

import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
cv2.imshow('Image', image)

# apply inverse thresholding
(T, threshInv) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Threshold Binary Inverse', threshInv)

# apply normal thresholding
(T, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold Binary', thresh)

cv2.imshow('Output', cv2.bitwise_and(image, image, mask=threshInv))  # a mask only considers pixels in the original image where the mask is greater than zero
cv2.waitKey(0)
