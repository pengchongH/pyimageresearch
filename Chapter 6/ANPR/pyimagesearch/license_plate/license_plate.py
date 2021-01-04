import numpy as np
import cv2
import imutils

class LicensePlateDetector:
    def __init__(self, image, minPlateW=60, minPlateH=20):
        # store image and the min width and height of the plate
        self.image = image
        self.minPlateW = minPlateW
        self.minPlateH = minPlateH

    def detect(self):
        return self.detectPlates()

    def detectPlates(self):
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        regions = []

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

        # find regions in the image that are light
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKernel)
        light = cv2.threshold(light, 50, 255, cv2.THRESH_BINARY)[1]

        # compute the scharr gradient representation of the blackhat image in the x-direction
        # and scale the resulting image into the range [0, 255].
        gradX = cv2.Sobel(blackhat, ddepth=cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype('uint8')

        # blur the gradX, apply a closing operation, threshold the image using Ostu's method
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # perform a series of erosions and dilations on the image
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # take a bitwise_and between the thresh and light, then erode and dilate
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)

        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            (w, h) = cv2.boundingRect(c)[2:]
            aspectRatio = w / float(h)

            # calculate percentage of area
            shapeArea = cv2.contourArea(c)
            bboxArea = w * h
            percentage = shapeArea / bboxArea

            # compute a rotated bounding box of the region
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.cv.Boxpoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect))

            if (aspectRatio > 3 and aspectRatio < 6) and h > self.minPlateH and w > self.minPlateW and percentage > 0.5:
                regions.append(box)

        return regions



