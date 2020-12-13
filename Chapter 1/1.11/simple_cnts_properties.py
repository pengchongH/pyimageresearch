import numpy as np
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', image)

# find external contours
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
clone = image.copy()

# Centroids
# loop the cnts
for c in cnts:
    # compute the moments of the contour which can be used to compute the centroid or "center of mass" of the region
    M = cv2.moments(c)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])

    # draw the center of the contour on the image
    cv2.circle(clone, (cX, cY), 10, (0, 255, 0), -1)

# show the output image
cv2.imshow('Centroids', clone)
cv2.waitKey(0)


# Area and Perimeter
# loop the cnts
clone = image.copy()
for (i, c) in enumerate(cnts):
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)  # A contour is closed if the shape outline is continuous and there are no holes
    print('Contour #{} --area: {:.2f}, perimeter: {:.2f}'.format(i + 1, area, perimeter))

    # draw the cnts on the image
    cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)

    # compute the center of the contour and draw the contour number
    M = cv2.moments(c)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    cv2.putText(clone, '#{}'.format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 4)

# show the output image
cv2.imshow('Contours', clone)
cv2.waitKey(0)


# Bounding box
clone = image.copy()
for c in cnts:
    # fit a bounding box to the contour
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the output image
cv2.imshow('Bounding Boxes', clone)
cv2.waitKey(0)


# Rotated Bounding Boxes
clone = image.copy()
for c in cnts:
    # fit a rotated bounding box to the contour and draw a rotated bounding box
    box = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(box))
    cv2.drawContours(clone, [box], -1, (0, 255, 0), 2)

# show the output image
cv2.imshow('Rotated Bounding Boxes', clone)
cv2.waitKey(0)


# Minimum Enclosing Circles
clone = image.copy()
for c in cnts:
    # fit a minimum enclosing circle to the contour
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    cv2.circle(clone, (int(x), int(y)), int(radius), (0, 255, 0), 2)

# show the output image
cv2.imshow('Min-Enclosing Circles', clone)
cv2.waitKey(0)


# Fitting an Ellipse
clone = image.copy()
for c in cnts:
    # to fit an ellipse, our contour must have at least 5 points
    if len(c) >= 5:
        # fit an ellipse to the contour
        ellipse = cv2.fitEllipse(c)
        cv2.ellipse(clone, ellipse, (0, 255, 0), 2)

# show the output image
cv2.imshow('Ellipses', clone)
cv2.waitKey(0)
