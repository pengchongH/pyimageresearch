from imutils import paths
import argparse
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--detector', required=True, help='path to trained object detector')
ap.add_argument('-t', '--testing', required=True, help='path to directory of testing images')
args = vars(ap.parse_args())

# load the detector
detector = dlib.simple_object_detector(args['detector'])

for testingPath in paths.list_images(args['testing']):
    image = cv2.imread(testingPath)
    boxes = detector(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    for b in boxes:
        (x, y, w, h) = (b.left(), b.top(), b.right(), b.bottom())
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

    cv2.imshow('Image', image)
    cv2.waitKey(0)
