# 1.compute the gradient orientation and magnitude
# 2.only display the pixels in the image that fall within the range min_{\theta} <= \theta <= max_{\theta}.

import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
ap.add_argument('-l', '--lower_angle', type=float, default=175.0, help='Lower orientation angle')
ap.add_argument('-u', '--upper_angle', type=float, default=180.0, help='Upper orientation angle')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', image)

# compute gradients along the X and Y
gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0)
gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1)

# compute the gradient magnitude and orientation respectively
mag = np.sqrt((gX ** 2) + (gY ** 2))
orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180   # % 180 to ensure the value of orientation is non-negative

# find all pixels that are within the upper and low angle boundaries
idxs = np.where(orientation >= args['lower_angle'], orientation, -1)
idxs = np.where(orientation <= args['upper_angle'], idxs, -1)
mask = np.zeros(gray.shape, dtype='uint8')
mask[idxs > -1] = 255

cv2.imshow('Mask', mask)
cv2.waitKey(0)
