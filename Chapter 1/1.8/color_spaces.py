# Color spaces: RGB, HSV, L*a*b* and grayscale

import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

# RGB(cube)
# additive color space
# the more of each color is added, the brighter the pixel becomes and the closer it comes to white
# USE:  display colors on a monitor

image = cv2.imread(args['image'])
cv2.imshow('RGB', image)

for (name, chan) in zip(['B', 'G', 'R'], cv2.split(image)):
    cv2.imshow(name, chan)

cv2.waitKey(0)
cv2.destroyAllWindows()


# HSV(cylinder)
# 3 dimensions: Hue, Saturation, Value
# note: different computer vision libraries will use different ranges(each dimension)
# USE: tracking the color of some object in an image

# convert the image to HSV color space and show it
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV', hsv)

for (name, chan) in zip(['H', 'S', 'V'], cv2.split(hsv)):
    cv2.imshow(name, chan)

cv2.waitKey(0)
cv2.destroyAllWindows()


# L*a*b*(3-axis system)
# mimic the methodology in which humans see and interpret color
# L-channel: The “lightness” of the pixel. This value goes up and down the vertical axis, white to black, with neutral grays at the center of the axis.
# a-channel: Originates from the center of the L-channel and defines pure green on one end of the spectrum and pure red on the other.
# b-channel: Originates from the center of the L-channel, but is perpendicular to the a-channel. The b-channel defines pure blue at one of the spectrum and pure yellow at the other.

# convert the image to L*a*b* color space and show it
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow('L*a*b*', lab)

for (name, chan) in zip(['L*', 'a*', 'b*'], cv2.split(lab)):
    cv2.imshow(name, chan)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Grayscale
# grayscale representation of a RGB image

# convert the image to grayscale and show it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale', gray)
cv2.imshow('Original', image)
cv2.waitKey(0)

# NOTE: when converting to grayscale, each RGB channel is not weighted UNIFORMLY
# Y = 0.299 * R + 0.587 * G + 0.114 * B
