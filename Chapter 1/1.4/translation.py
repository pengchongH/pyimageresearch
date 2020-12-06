# NOTE: Translating (shifting) an image is given by a NumPy matrix in the form:
# [[1, 0, tX], [0, 1, tY]]
# Negative values of tX will shift the image to the left and positive values will shift the image to the right.
# Negative values of tY will shift the image up and positive values will shift the image down.
import numpy as np
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)

# translate the image 25 pixels to the right and 50 pixels down
Mat = np.float32([[1, 0, 25], [0, 1, 50]])
shifted = cv2.warpAffine(image, Mat, (image.shape[1], image.shape[0]))  # (width, height)
cv2.imshow('Shifted Down and Right', shifted)

# translate the image 50 pixels to the left and 90 pixels up
Mat = np.float32([[1, 0, -50], [0, 1, -90]])
shifted = cv2.warpAffine(image, Mat, (image.shape[1], image.shape[0]))  # (width, height)
cv2.imshow('Shifted Up and Left', shifted)

# USE: imutils.translate
shifted = imutils.translate(image, 0, 100)
cv2.imshow('Shifted Down', shifted)
cv2.waitKey(0)

