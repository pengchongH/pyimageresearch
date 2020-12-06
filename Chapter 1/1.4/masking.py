# NOTE: bitwise & masking
# A mask allows us to focus only on
# the portions of the image that interests us.

import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)

# construct a rectangular mask
mask = np.zeros(image.shape[:2], dtype='uint8')
cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)
cv2.imshow('Mask', mask)
masked = cv2.bitwise_and(image, image, mask=mask)   # NOTE: mask keyword supplied, only pixels that are part of the white rectangle.
cv2.imshow('Mask Applied', masked)
cv2.waitKey(0)

# construct a circular mask
mask = np.zeros(image.shape[:2], dtype='uint8')
cv2.circle(mask, (145, 200), 100, 255, -1)
cv2.imshow('Mask', mask)
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow('Mask Applied', masked)
cv2.waitKey(0)