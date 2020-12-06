# NOTE:  theta, the number of (counter-clockwise) degrees we are going to rotate the image by.

import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)

(height, width) = image.shape[:2]
(cX, cY) = (width // 2, height // 2)

# rotate image by 45 degrees
Mat = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
rotated = cv2.warpAffine(image, Mat, (width, height))
cv2.imshow('Rotated by 45 Degrees', rotated)

# rotate image by -90 degrees
Mat = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
rotated = cv2.warpAffine(image, Mat, (width, height))
cv2.imshow('Rotated by -90 Degrees', rotated)

# rotate image around an arbitrary point rather than the center
Mat = cv2.getRotationMatrix2D((cX - 50, cY - 50), -90, 1.0)
rotated = cv2.warpAffine(image, Mat, (width, height))
cv2.imshow('Rotated by -90&new—center Degrees', rotated)

# USE: imutils.rotate
rotated = imutils.rotate(image, 180, (cX, cY))
cv2.imshow("Rotated by 180 Degrees", rotated)
cv2.waitKey(0)

