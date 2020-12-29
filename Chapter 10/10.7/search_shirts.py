from __future__ import print_function
from pyimagesearch import LocalBinaryPatterns
from imutils import paths
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to the dataset')
ap.add_argument('-q', '--query', required=True, help='path to the query image')
args = vars(ap.parse_args())

desc = LocalBinaryPatterns(24, 8)
index = {}

for imagePath in paths.list_images(args['dataset']):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)

    filename = imagePath[imagePath.rfind('/') + 1:]
    index[filename] = hist

query = cv2.imread(args['query'])
queryFeatures = desc.describe(cv2.cvtColor(query, cv2.COLOR_BGR2GRAY))

cv2.imshow('Query', query)
results = {}

for (k, features) in index.items():
    d = 0.5 * np.sum(((features - queryFeatures) ** 2) / (features + queryFeatures + 1e-10))
    results[k] = d

results = sorted([(v, k) for (k, v) in results.items()])[:3]

for (i, (score, filename)) in enumerate(results):
    print('#%d. %s: %.4f' % (i + 1, filename, score))
    image = cv2.imread(args['dataset'] + '/' + filename)
    cv2.imshow('Result #{}'.format(i + 1), image)
    cv2.waitKey(0)