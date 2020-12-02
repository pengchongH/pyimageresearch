
# NOTE! OpenCV stores images in BGR order rather than RGB order

# import packages
import argparse
import cv2

# parse argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image')
args = vars(ap.parse_args())

# load the image, dimensions, display
image = cv2.imread(args['image'])
(height, width) = image.shape[:2]
cv2.imshow('Original', image)

(b, g, r) = image[0, 0]
print('pixel at (0, 0) - Red: {r}, Green: {g}, Blue: {b}'.format(r=r, g=g, b=b))

# change the value of pixel at (0, 0) to red
image[0, 0] = (0, 0, 255)
(b, g, r) = image[0, 0]
print('pixel at (0, 0) - Red: {r}, Green: {g}, Blue: {b}'.format(r=r, g=g, b=b))

# rectangular region
(cX, cY) = (width // 2, height // 2)
tl = image[0:cY, 0:cX]
tr = image[0:cY, cX:width]
bl = image[cY:height, 0:cX]
br = image[cY:height, cX:width]
cv2.imshow('Top-Left Corner', tl)
cv2.imshow("Top-Right Corner", tr)
cv2.imshow("Bottom-Right Corner", br)
cv2.imshow("Bottom-Left Corner", bl)

# tl to green
image[0:cY, 0:cX] = (0, 255, 0)

# show updated image
cv2.imshow('Updated', image)
cv2.waitKey(0)