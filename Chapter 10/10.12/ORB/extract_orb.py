import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detector = cv2.ORB_create()
(kps, descs) = detector.detectAndCompute(gray, None)

print('[INFO] # of the keypoints detected: {}'.format(len(kps)))
print('[INFO] feature vector shape: {}'.format(descs.shape))


