import imutils
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

wide = cv2.Canny(blurred, 10, 200)
tight = cv2.Canny(blurred, 225, 250)
# USE: imutils.auto_canny()
auto = imutils.auto_canny(blurred)

# show the edge maps
cv2.imshow('Original', image)
cv2.imshow('Wide edge map', wide)
cv2.imshow('Tight edge maps', tight)
cv2.imshow('Auto edge maps', auto)
cv2.waitKey(0)