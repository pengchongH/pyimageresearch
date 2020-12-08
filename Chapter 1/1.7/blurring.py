# Blurring
# use a low-pass filter to reduce the amount of noise and detail in an image
# one of the most common pre-processing steps in computer vision and image processing
# Note: thresholding and edge detection, perform better if the image is first smoothed or blurred

# OPTIONS: averaging; Gaussian blurring; median filtering; bilateral filtering

# averaging
# RULE: the larger your smoothing kernel is, the more blurred your image will look.
# kernel: rectangular

import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)
kernelSizes = [(3, 3), (9, 9), (15, 15)]

# apply an 'average' blur to the image
for kernelSize in kernelSizes:
    blurred = cv2.blur(image, kernelSize)
    cv2.imshow('Average ({}, {})'.format(kernelSize[0], kernelSize[1]), blurred)
    cv2.waitKey(0)


# Gaussian blurring
# weighted mean
# preserve more of the edges in the image as compared to average smoothing
# kernel: rectangular
cv2.destroyAllWindows()
cv2.imshow('Original', image)

# apply a 'Gaussian' blur to the image
for kernelSize in kernelSizes:
    blurred = cv2.GaussianBlur(image, kernelSize, 0)
    cv2.imshow('Gaussian ({}, {})'.format(kernelSize[0], kernelSize[1]), blurred)
    cv2.waitKey(0)


# median filtering
# removing salt-and-pepper noise effectively
# kernel: square
cv2.destroyAllWindows()
cv2.imshow('Original', image)

# apply a 'median filtering' to the image
for k in (3, 9, 15):
    blurred = cv2.medianBlur(image, k)
    cv2.imshow('Median {}'.format(k), blurred)
    cv2.waitKey(0)


# bilateral blurring
# reduce noise while still maintaining edges
# by introducing two Gaussian distributions
# Disadvantage: slower than averaging, Gaussian, and median blurring counterparts.
cv2.destroyAllWindows()
cv2.imshow('Original', image)

# apply a 'bilateral' blur to the image
# params
# diameter: diameter of pixel neighborhood
# sigmaColor: color standard deviation(A larger value for sigmaColor means that more colors in the neighborhood will be considered when computing the blur.)
# sigmaSpace: space standard deviation(A larger value of sigmaSpace means that pixels farther out from the central pixel diameter will influence the blurring calculation.)
params = [(11, 21, 7), (11, 41, 21), (11, 61, 39)]
for (diameter, sigmaColor, sigmaSpace) in params:
    blurred = cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)
    title = 'Blurred d={}, sc={}, ss={}'.format(diameter, sigmaColor, sigmaSpace)
    cv2.imshow(title, blurred)
    cv2.waitKey(0)





