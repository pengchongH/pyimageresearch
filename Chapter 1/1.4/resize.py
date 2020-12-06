# NOTE: aspect ratio = width / height

import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)

# we need to keep in mind aspect ratio so the image does not look skewed
# or distorted -- therefore, we calculate the ratio of the new image to
# the old image. Let's make our new image have a width of 150 pixels
ratio = 150.0 / image.shape[1]
dim = (150, int(image.shape[0] * ratio))  # NOTE:dim[0], dim[1] are int datatype

# resizing
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('Resized(width)', resized)

# what if we wanted to adjust the height of the image? We can apply
# the same concept, again keeping in mind the aspect ratio, but instead
# calculating the ratio based on height -- let's make the height of the
# resized image 50 pixels

ratio = 50.0 / image.shape[0]
dim = (int(image.shape[1] * ratio), 50)

# resizing
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('Resized(height)', resized)
cv2.waitKey(0)

# USE: imutils.translate
resized = imutils.resize(image, width=100)
cv2.imshow('Resized via imutils', resized)
cv2.waitKey(0)

# which interpolation method?
# increasing: cv2.INTER_LINEAR cv2.INTER_CUBIC
# decreasing: cv2.INTER_AREA

# NOTE:
# default :cv2.INTER_LINEAR
# provides the highest quality results at a modest computation cost.

