from __future__ import print_function
from pyimagesearch.license_plate import LicensePlateDetector
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images', required=True, help='path to the iamges')
args = vars(ap.parse_args())

for imagePath in sorted(list(paths.list_images(args['images']))):
    image = cv2.imread(imagePath)
    print(imagePath)

    # ensure the width is greater than 640 pixels
    if image.shape[1] > 640:
        image = imutils.resize(image, width=640)

    lpd = LicensePlateDetector(image)
    plates = lpd.detect()

    for lpBox in plates:
        lpBox = np.array(lpBox).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(image, [lpBox], -1, (0, 255, 0), 2)
        cv2.imshow('image', image)
        cv2.waitKey(0)