# Otsu’s Method
# assumes that image contains two classes of pixels: the background and the foreground.
# the histogram has two peaks
# global thresholding — implying that a single value of T is computed for the entire image.


import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
cv2.imshow('Image', image)

# apply Otsu's thresholding to the image
(T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow('Threshold', threshInv)
print("Otsu's thresholding value: {}".format(T))

cv2.imshow('Output', cv2.bitwise_and(image, image, mask=threshInv))
cv2.waitKey(0)