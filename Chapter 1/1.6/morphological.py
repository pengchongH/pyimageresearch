# Objective:
# Erosion
# Dilation
# Opening
# Closing
# Morphological gradient
# Black hat
# Top hat (White hat)

import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', image)

# Structuring element
# A structuring element can be viewed as a type of kernel or mask
# cv2.getStructuringElement()

# Erosion
# A foreground pixel in the input image will be kept only if ALL pixels inside the structuring element are > 0.
# Otherwise, the pixels are set to 0 (i.e. background).
# USE: removing small blobs in an image or disconnecting two connected objects

# apply a series of erosion
for i in range(0, 3):
    eroded = cv2.erode(gray.copy(), None, iterations=i + 1)  # kernel = None means  a 3 * 3  8-neighborhood s_element
    cv2.imshow('Eroded {} times'.format(i + 1), eroded)
    cv2.waitKey(0)

# Dilation
# opposite to erosion
# a center pixel p of the structuring element is set to white if ANY pixel in the structuring element is > 0
# USE: joining broken parts of an image together
cv2.destroyAllWindows()
cv2.imshow('Original', image)

# apply a series of dilation
for i in range(0, 3):
    dilated = cv2.dilate(gray.copy(), None, iterations=i + 1)
    cv2.imshow('Dilated {} times'.format(i + 1), dilated)
    cv2.waitKey(0)

# Opening
# erosion followed by a dilation
# USE: removing small blobs from an image
cv2.destroyAllWindows()
cv2.imshow('Original', image)
kernelSizes = [(3, 3),  (5, 5), (7, 7)]  # defines the width and height respectively of the structuring element

# apply 'opening' operation
for kernelSizes in kernelSizes:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSizes)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    cv2.imshow('Opening: ({}, {})'.format(kernelSizes[0], kernelSizes[1]), opening)
    cv2.waitKey(0)

# Closing
# dilation followed by an erosion
# USE: closing holes inside of objects or connecting components together
cv2.destroyAllWindows()
cv2.imshow('Original', image)

# apply 'closing' operation
for kernelSizes in kernelSizes:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSizes)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Closing: ({}, {})'.format(kernelSizes[0], kernelSizes[1]), closing)
    cv2.waitKey(0)

# Morphological Gradient
# USE: determining the outline of a particular object of an image
cv2.destroyAllWindows()
cv2.imshow('Original', image)

# apply 'morphological gradient' operation
for kernelSizes in kernelSizes:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSizes)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow('Gradient: ({}, {})'.format(kernelSizes[0], kernelSizes[1]), gradient)
    cv2.waitKey(0)

