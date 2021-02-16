from pyimagesearch.object_detection.helpers import pyramid
import argparse
import cv2
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
ap.add_argument('-s', '--scale', required=True, default=1.5, help='scale factor size')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])

for (i, layer) in enumerate(pyramid(image, scale=float(args['scale']))):
    cv2.imshow('Layer {}'.format(i + 1), layer)
    cv2.waitKey(0)

