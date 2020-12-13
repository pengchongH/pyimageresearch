# NOTE: switch from grayscale to binary images to improve contour accuracy

import numpy as np
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', image)

# find contours and draw
cnts = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # cv2.findContours function is destructive to the input image
cnts = imutils.grab_contours(cnts)
clone = image.copy()
cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
print('Found {} contours'.format(len(cnts)))

# show the output image
cv2.imshow('All Contours', clone)
cv2.waitKey(0)


# draw contours individually
# re-clone the image and destroy all windows
clone = image.copy()
cv2.destroyAllWindows()

# loop over the contours individually and draw
for (i, c) in enumerate(cnts):
    print('Drawing contour #{}'.format(i + 1))
    cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)
    cv2.imshow('Single Contour', clone)
    cv2.waitKey(0)


# find and draw only EXTERNAL contours
# re-clone the image and destroy all windows
clone = image.copy()
cv2.destroyAllWindows()

# find the EXTERNAL contours in the image
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
print('Found {} EXTERNAL contours'.format(len(cnts)))

# show the output image
cv2.imshow('ALL CONTOURS', clone)
cv2.waitKey(0)


# using both contours and masks together
# re-clone the image and destroy all windows
clone = image.copy()
cv2.destroyAllWindows()

# loop over the contours individually, draw a mask for the contour, and then apply a bitwise AND
for c in cnts:
    # construct a mask
    mask = np.zeros(gray.shape, dtype='uint8')
    cv2.drawContours(mask, [c], -1, 255, -1)

    # show the image
    cv2.imshow('Image', image)
    cv2.imshow('Mask', mask)
    cv2.imshow('Image + Mask', cv2.bitwise_and(image, image, mask=mask))
    cv2.waitKey(0)