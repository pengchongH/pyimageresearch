import imutils
import cv2

image = cv2.imread('planes.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Hu moments of entire image
moments = cv2.HuMoments(cv2.moments(image)).flatten()
print('Original moment: {}'.format(moments))
cv2.imshow('Image', image)
cv2.waitKey(0)

# Hu moments of the three planes
# REMEMBER: keep in mind the number of objects in an image
cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = image[y:y + h, x:x + w]
    moments = cv2.HuMoments(cv2.moments(roi)).flatten()

    print('moment for plane #{}: {}'.format(i + 1, moments))
    cv2.imshow('ROI #{}'.format(i + 1), roi)
    cv2.waitKey(0)

