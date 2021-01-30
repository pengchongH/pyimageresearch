import cv2
import imutils

class DistanceFinder:
    def __init__(self, knownWidth, knownDistance):
        self.knownWidth = knownWidth
        self.knownDistance = knownDistance

        self.focalLength = 0

    def calibrate(self, width):
        self.focalLength = (self.knownDistance * width) / self.knownWidth

    def distance(self, perceiveWidth):
        return (self.knownWidth * self.focalLength) / perceiveWidth

    @staticmethod
    def findSquareMarker(image):
        # cvt grayscale, blur and find edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray_blurred, 35, 125)

        # find cnts and sort according their area (largest to smallest)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        markerDim = None

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # ensure the cnt is a rectangle
            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                aspectRatio = w / float(h)

                # check if the ratio is in the bounds
                if aspectRatio > 0.9 and aspectRatio < 1.1:
                    markerDim = (x, y, w, h)
                    break

        return markerDim

    @staticmethod
    def draw(image, boundingBox, dist, color=(0, 255, 0), thickness=2):
        (x, y, w, h) = boundingBox
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(image, '%.2fft' % (dist / 12), (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)

        return image



