from pyimagesearch.markers import DistanceFinder
from imutils import paths
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-r', '--reference', required=True, help='path to the reference image')
ap.add_argument('-w', '--ref_width_inches', required=True, type=float, help='reference object width in inches')
ap.add_argument('-d', '--ref_distance_inches', required=True, type=float, help='distance to reference object in inches')
ap.add_argument('-i', '--image', required=True, help='path to the directory containing images to test')
args = vars(ap.parse_args())

refImage = cv2.imread(args['reference'])
refImage = imutils.resize(refImage, height=700)

# initialize the distance finder
df = DistanceFinder(args['ref_width_inches'], args['ref_distance_inches'])
refMarker = DistanceFinder.findSquareMarker(refImage)
df.calibrate(refMarker[2])

# visualize the results about refImage
refImage = df.draw(refImage, refMarker, df.distance(refMarker[2]))
cv2.imshow('Reference', refImage)

for imagePath in paths.list_images(args['image']):
    filename = imagePath[imagePath.rfind('/') + 1:]
    image = cv2.imread(imagePath)
    image = imutils.resize(image, height=700)
    print('[INFO] processing {}'.format(filename))

    marker = DistanceFinder.findSquareMarker(image)

    if marker is None:
        print('[INFO] could not find marker for {}'.format(filename))
        continue

    distance = df.distance(marker[2])

    image = df.draw(image, marker, distance)
    cv2.imshow('image', image)
    cv2.waitKey(0)


