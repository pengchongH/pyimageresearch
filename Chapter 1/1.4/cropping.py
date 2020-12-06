# NOTE: Cropping, Region of Interest(ROI)

import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)

# crop the face from the image(florida_trip.png)
face = image[85:250, 85:220]  # NOTE: image[height, width]
cv2.imshow("Face", face)
cv2.waitKey(0)

# crop the entire body
body = image[90:450, 0:290]
cv2.imshow("Body", body)
cv2.waitKey(0)

