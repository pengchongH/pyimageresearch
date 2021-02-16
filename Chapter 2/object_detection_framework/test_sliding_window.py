from pyimagesearch.object_detection.helpers import sliding_window
from pyimagesearch.object_detection.helpers import pyramid
import argparse
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
ap.add_argument('-w', '--width', required=True, type=int, help='width of sliding window')
ap.add_argument('-t', '--height', required=True, type=int, help='height of sliding window')
ap.add_argument('-s', '--scale', type=float, default=1.5, help='scale factor size')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
(winW, winH) = (args['width'], args['height'])

for layer in pyramid(image, scale=args['scale']):
    for (x, y, window) in sliding_window(layer, stepSize=32, windowSize=(winW, winH)):
        # if the current window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        clone = layer.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow('window', clone)

        cv2.waitKey(1)
        time.sleep(0.025)

