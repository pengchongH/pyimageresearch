import numpy as np
import argparse
import cv2
from imutils.feature.factories import FeatureDetector_create, DescriptorExtractor_create, DescriptorMatcher_create

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--first', required=True, help='path to the first image')
ap.add_argument('-s', '--second', required=True, help='path to the second image')
ap.add_argument('-d', '--detector', type=str, default='SURF', help='Keypoint detector to use. Options ["BRISK", "DENSE", "DOG", "SIFT", "FAST", "FASTHESSIAN", "SURF", "GFTT", "HARRIS", "MESR", "ORB", "STAR"]')
ap.add_argument('-e', '--extractor', type=str, default='SIFT', help='Feature extractor to use. Options ["RootSIFT", "SIFT", "SURF"]')
ap.add_argument('-m', '--matcher', type=str, default='BruteForce', help='Feature matcher to use. Options ["BruteForce", "BruteForce-SL2", "BruteForce-L1", "FlannBased"]')
ap.add_argument('-v', '--visualize', type=str, default='Yes', help='Whether the visualiztion image should be shown. Options ["Yes", "No", "Each"]')
args = vars(ap.parse_args())

# initialize the feature detector
if args['detector'] == 'DOG':
    detector = FeatureDetector_create('SIFT')
elif args['detector'] == 'FASTHESSIAN':
    detector = FeatureDetector_create('SURF')
else:
    detector = FeatureDetector_create(args['detector'])

# initialize the feature extractor
extractor = DescriptorExtractor_create(args['extractor'])

# initialize the keypoint matcher
matcher = DescriptorMatcher_create(args['matcher'])

# load the two image and cvt grayscale
imageA = cv2.imread(args['first'])
imageB = cv2.imread(args['second'])
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# detect keypoints
kpsA = detector.detect(grayA)
kpsB = detector.detect(grayB)

# extract features from each of the keypoints regions in the images
(kpsA, featureA) = extractor.compute(grayA, kpsA)
(kpsB, featureB) = extractor.compute(grayB, kpsB)

# match the keypoints (using the Euclidean distance)
rawMatches = matcher.knnMatch(featureA, featureB, 2)
matches = []

if rawMatches is not None:
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.8:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    print('# of keypoints from first image: {}'.format(len(kpsA)))
    print('# of keypoints from second image: {}'.format(len(kpsB)))
    print('# of matched keypoints: {}'.format(len(matches)))

    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype='uint8')
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    for (trainIdx, queryIdx) in matches:
        color = np.random.randint(0, high=255, size=(3,))
        color = tuple(map(int, color))
        ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
        ptB = (int(kpsB[trainIdx].pt[0] + wA), int(kpsB[trainIdx].pt[1]))
        cv2.line(vis, ptA, ptB, color, 2)

        if args['visualize'] == 'Each':
            cv2.imshow('Matched', vis)
            cv2.waitKey(0)

    if args['visualize'] == 'Yes':
        cv2.imshow('Matched', vis)
        cv2.waitKey(0)




