import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-s', '--source', required=True, help='path to the source image')
ap.add_argument('-t', '--template', required=True, help='path to the template image')
args = vars(ap.parse_args())

source = cv2.imread(args['source'])
template = cv2.imread(args['template'])
(tempW, tempH) = template.shape[:2]

result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF)
(minVal, maxVal, minLoc, (x, y)) = cv2.minMaxLoc(result)

cv2.rectangle(source, (x, y), (x + tempW, y + tempH), (0, 255, 0), 2)

cv2.imshow('Source', source)
cv2.imshow('Template', template)
cv2.waitKey(0)

