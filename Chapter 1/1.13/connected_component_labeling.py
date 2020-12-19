# connected-component analysis: binary or thresholded image

# The first pass

# step 1: check if we care about the central pixel p or not
# if p == 0:
#   ignore
# else:
#   proceed to step 2 and step 3

# step 2 and step 3
# north and west pixels, denoted as N and W
# if N and W are background pixels:
#   create a new label
# elif N and/or W are not background pixels:
#   proceed to step 4 and step 5

# step 4 and step 5
# label of center pixel p = min(N, M)

# step 6
# union-find data structure

# step 7
# Continue to the next pixel and go repeat the process beginning with Step 1


# The second pass
# looping over the image once again, one pixel at a time
# if label in a set, they have common minimum label


from skimage.filters import threshold_local
from skimage import measure
import numpy as np
import cv2

# load image
plate = cv2.imread('license_plate.png')

# convert to HSV and apply adaptive thresholding
V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
thresh = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 3)

cv2.imshow('License plate', plate)
cv2.imshow('Thresh', thresh)
cv2.waitKey(0)

# apply connected-component analysis
labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype='uint8')
print('[INFO] found {} blobs'.format(len(np.unique(labels))))

for (i, label) in enumerate(np.unique(labels)):
    if label == 0:
        print('[INFO] label: 0(background)')
        continue

    print('[INFO] label: {}(foreground)'.format(i))
    labelMask = np.zeros(thresh.shape, dtype='uint8')
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)

    # ensure the blobs be the useful information
    if numPixels > 300 and numPixels < 1500:
        mask = cv2.add(mask, labelMask)

    cv2.imshow('Label', labelMask)
    cv2.waitKey(0)

cv2.imshow('Large blobs', mask)
cv2.waitKey(0)