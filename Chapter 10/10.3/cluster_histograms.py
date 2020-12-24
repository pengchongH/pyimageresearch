from pyimagesearch.descriptors.labhistogram import LabHistogram
from sklearn.cluster import KMeans
from imutils import paths
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to the input dataset directory')
ap.add_argument('-k', '--clusters', type=int, default=2, help='# of clusters to generate')
args = vars(ap.parse_args())

# initialize the image descriptor along with the image mat
desc = LabHistogram([8, 8, 8])
data = []

imagePaths = list(paths.list_images(args['dataset']))
imagePaths = np.array(sorted(imagePaths))

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    hist = desc.describe(image)
    data.append(hist)

# cluster the color histogram
clt = KMeans(n_clusters=args['clusters'])
labels = clt.fit_predict(data)

for label in np.unique(labels):
    labelPaths = imagePaths[np.where(labels == label)]

    for (i, path) in enumerate(labelPaths):
        image = cv2.imread(path)
        cv2.imshow('Cluster {}, Image #{}'.format(label + 1, i + 1), image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




