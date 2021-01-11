import numpy as np
import cv2
import imutils

image = cv2.imread('grand_central_terminal.png')
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detector = cv2.FastFeatureDetector_create()
kps = detector.detect(gray)

print('# of the keypoints: {}'.format(len(kps)))

for kp in kps:
    r = int(0.5 * kp.size)
    (x, y) = np.int0(kp.pt)
    cv2.circle(image, (x, y), r, (0, 255, 255), 2)

cv2.imshow('Image', np.hstack([orig, image]))
cv2.waitKey(0)