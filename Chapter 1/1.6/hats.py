# White Hat
# USE: reveal bright regions of an image on dark backgrounds
# Black Hat
# USE: reveal dark regions (i.e. the license plate text) against light backgrounds (i.e. the license plate itself).
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', image)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))  # a license plate is roughly 3x wider than it is tall
whitehat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

cv2.imshow('Original', image)
cv2.imshow('Blackhat', blackhat)
cv2.imshow('Tophat', whitehat)
cv2.waitKey(0)
