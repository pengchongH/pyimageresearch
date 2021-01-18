import numpy as np
import argparse
import cv2
import imutils

def dense(image, step, radius):
    # initialize the list of kps
    kps = []

    for x in range(0, image.shape[1], step):
        for y in range(0, image.shape[0], step):
            kps.append(cv2.KeyPoint(x, y, radius))

    return kps

ap = argparse.ArgumentParser()
ap.add_argument('-s', '--step', type=int, default=6, help='step(in pixels) of the dense detector')
ap.add_argument('-r', '--size', type=int, default=1, help='default diameter of keypoint')
args = vars(ap.parse_args())

image = cv2.imread('next.png')
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kps = dense(gray, args['step'], args['size'] / 2)

print('# of the keypoints: {}'.format(len(kps)))

for kp in kps:
    kp.size = args['size']

for kp in kps:
    r = int(0.5 * kp.size)
    (x, y) = np.int0(kp.pt)
    cv2.circle(image, (x, y), r, (0, 255, 255), 2)

cv2.imshow('Image', np.hstack([orig, image]))
cv2.waitKey(0)


