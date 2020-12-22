# image descriptor
# Step 1: Separate the input image into its respective channels.
# Step 2: Compute various statistics for each channel, such as mean, standard deviation, skew, and kurtosis.
# Step 3: Concatenate the statistics together to form a “list”(feature vector) of statistics for each color channel.

from scipy.spatial import distance as dist
from imutils import paths
import numpy as np
import cv2

image_paths = sorted(list(paths.list_images('pictures')))
index = {}

for image_path in image_paths:
    image = cv2.imread(image_path)
    filename = image_path[image_path.rfind('/') + 1:]

    (means, stds) = cv2.meanStdDev(image)
    features = np.concatenate([means, stds]).flatten()
    index[filename] = features

# display a query image and sort the key value of the dictionary
query = cv2.imread(image_paths[0])
cv2.imshow('Query 01', query)
keys = sorted(index.keys())

for (i, k) in enumerate(keys):
    if k == 'trex_01.png':
        continue

    image = cv2.imread(image_paths[i])
    d = dist.euclidean(index['trex_01.png'], index[k])

    cv2.putText(image, '{:.2f}'.format(d), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow(k, image)

cv2.waitKey(0)

