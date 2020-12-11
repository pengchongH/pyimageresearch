# Canny edge detection algorithm
# 1.Applying Gaussian smoothing to the image to help reduce noise.
# 2.Computing the G_{x} and G_{y} image gradients using the Sobel kernel.
# 3.Applying non-maxima suppression to keep only the local maxima of gradient magnitude pixels that are pointing in the direction of the gradient.
# 4.Defining and applying the T_{upper} and T_{lower}

import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

cv2.imshow('Original', image)
cv2.imshow('Blurred', blurred)

# compute a 'wide', 'mid-range' and 'tight' threshold for the edges
wide = cv2.Canny(blurred, 10, 200)
mid = cv2.Canny(blurred, 30, 150)
tight = cv2.Canny(blurred, 240, 250)

# show the edge maps
cv2.imshow('Wide edge map', wide)
cv2.imshow('Mid edge map', mid)
cv2.imshow('Tight edge maps', tight)
cv2.waitKey(0)