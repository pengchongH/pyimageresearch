from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import argparse
import glob
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to the dataset directory')
args = vars(ap.parse_args())

# grab the image paths from disk and initialize the data matrix
imagePaths = sorted(glob.glob(args['dataset'] + '/*.jpg'))
data = []

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    (x, y, w, h) = cv2.boundingRect(c)
    roi = cv2.resize(thresh[y:y + h, x:x + w], (50, 50))
    moments = cv2.HuMoments(cv2.moments(roi)).flatten()
    data.append(moments)

D = pairwise_distances(data).sum(axis=1)
i = np.argmax(D)

image = cv2.imread(imagePaths[i])
print('Found square: {}'.format(imagePaths[i]))
cv2.imshow('Outlier', image)
cv2.waitKey(0)


