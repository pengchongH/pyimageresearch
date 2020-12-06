# NOTE: AND, OR, XOR, NOT
# cv2.bitwise_and()
# cv2.bitwise_or()
# cv2.bitwise_xor()
# cv2.bitwise_not()

import numpy as np
import cv2

# draw a rectangle
rectangle = np.zeros((300, 300), dtype='uint8')
cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
cv2.imshow('Rectangle', rectangle)

# draw circle
circle = np.zeros((300, 300), dtype='uint8')
cv2.circle(circle, (150, 150), 150, 255, -1)
cv2.imshow('Circle', circle)

# AND
bitwised = cv2.bitwise_and(rectangle, circle)
cv2.imshow('AND', bitwised)
cv2.waitKey(0)

# OR
bitwised = cv2.bitwise_or(rectangle, circle)
cv2.imshow('OR', bitwised)
cv2.waitKey(0)

# XOR
bitwised = cv2.bitwise_xor(rectangle, circle)
cv2.imshow('XOR', bitwised)
cv2.waitKey(0)

# NOT
bitwised = cv2.bitwise_not(circle)
cv2.imshow('NOT', bitwised)
cv2.waitKey(0)
