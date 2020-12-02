# Draw lines, rectangles and circles with openCV

import numpy as np
import cv2

# initialize a canvas
canvas = np.zeros((300, 300, 3), dtype='uint8')
test = np.zeros((5, 5, 3), dtype='uint8')
# draw a green line from the top-left corner of canvas to the bottom-right
green = (0, 255, 0)
cv2.line(canvas, (0, 0), (300, 300), green)
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)

# draw a 3 pixel thick red line from the top-right corner to the bottom-left
red = (0, 0, 255)
cv2.line(canvas, (300, 0), (0, 300), red, 3)  # thickness = 3
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)

# draw a green 50x50 pixel square, starting at 10x10 and ending at 60x60
cv2.rectangle(canvas, (10, 10), (60, 60), green)  # (10, 10), (59,59)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# draw a red rectangle, thickness = 5
cv2.rectangle(canvas, (50, 200), (200, 225), red, 5)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# draw a blue rectangle and full of blue
blue = (255, 0, 0)
cv2.rectangle(canvas, (200, 50), (225, 125), blue, -1)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# reset canvas and draw a white circle at the center of canvas
canvas = np.zeros((300, 300, 3), dtype="uint8")
(c_x, c_y) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
white = (255, 255, 255)

for r in range(0, 175, 25):
    cv2.circle(canvas, (c_x, c_y), r, white)

cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# draw 25 random circles
for i in range(0, 25):
    # randomly generate a radius size between 5 and 200, generate a random
    # color, and then pick a random point on our canvas where the circle
    # will be drawn
    radius = np.random.randint(5, 201)
    color = np.random.randint(0, 256, size=(3,)).tolist()
    point = np.random.randint(0, 300, size=(2,))
    cv2.circle(canvas, tuple(point), radius, tuple(color), -1)

cv2.imshow("Canvas", canvas)
cv2.waitKey(0)
