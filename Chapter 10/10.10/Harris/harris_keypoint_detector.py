import numpy as np
import cv2
import imutils

def harris(gray, blockSize=2, apetureSize=3, k=0.1, T=0.02):
    gray = np.float32(gray)
    H = cv2.cornerHarris(gray, blockSize, apetureSize, k)

    kps = np.argwhere(H > T * H.max())
    kps = [cv2.KeyPoint(pt[1], pt[0], 3) for pt in kps]

    return kps

image = cv2.imread('next.png')
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kps = harris(gray)
print('# of the keypoints: {}'.format(len(kps)))

for kp in kps:
    r = int(0.5 * kp.size)
    (x, y) = np.int0(kp.pt)
    cv2.circle(image, (x, y), r, (0, 255, 255), 2)

cv2.imshow('Image', np.hstack([orig, image]))
cv2.waitKey(0)