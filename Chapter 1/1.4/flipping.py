# NOTE: cv2.flip(image, flipcode)
# flipcode = 0, vertically
# flipcode = 1, horizontally
# flipcode = -1, both axes

import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)

# flip horizontally
flipped = cv2.flip(image, 1)
cv2.imshow('Flipped Horizontally', flipped)

# flipped vertically
flipped = cv2.flip(image, 0)
cv2.imshow('Flipped Vertically', flipped)

# flipped both axes
flipped = cv2.flip(image, -1)
cv2.imshow('Flipped Horizontally&Vertically', flipped)
cv2.waitKey(0)


# QUIZ code
# flipped = cv2.flip(image, 1)
# rotated = imutils.rotate(flipped, 45)
# flipped = cv2.flip(rotated, 0)
# print(flipped[189, 441])