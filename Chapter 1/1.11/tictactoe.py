import cv2
import imutils

image = cv2.imread('tictactoe.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for (i, c) in enumerate(cnts):
    # compute the area of contour and bounding bax to compute the aspect ratio
    area = cv2.contourArea(c)
    (x, y, w, h) = cv2.boundingRect(c)

    # compute solidity
    hull = cv2.convexHull(c)
    hullArea = cv2.contourArea(hull)
    solidity = area / hullArea

    # initialize the character text
    char = '?'

    # if the solidity is high, then we are examining an 'O'
    if solidity > 0.9:
        char = 'O'

    # otherwise, if the solidity it still reasonably high, we are examining an 'X'
    elif solidity > 0.5:
        char = 'X'

    # if the character is not unknown, draw it
    if char != '?':
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.putText(image, char, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 4)

    # show the contour properties
    print('{}(Contour #{}) --  solidity={:.2f}'.format(char, i + 1, solidity))

# show the output image
cv2.imshow('Output', image)
cv2.waitKey(0)

