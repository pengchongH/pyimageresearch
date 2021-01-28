from pyimagesearch.descriptors import RootSIFT
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detector = cv2.xfeatures2d.SIFT_create()
extractor = RootSIFT()

#(kps, _) = detector.detectAndCompute(gray, None)
(kps, descs) = extractor.compute(gray, None)

print('[INFO] # of keypoints detected: {}'.format(len(kps)))
print('[INFO] feature vector shape: {}'.format(descs.shape))


