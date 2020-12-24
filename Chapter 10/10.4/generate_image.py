import numpy as np
import argparse
import uuid
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='path to the output directory')
ap.add_argument('-n', '--num_image', type=int, default=500, help='# of distractor images to generate')
args = vars(ap.parse_args())

for i in range(0, args['num_image']):
    image = np.zeros((500, 500, 3), dtype='uint8')
    (x, y) = np.random.uniform(105, 395, size=(2,)).astype('int0')
    r = np.random.uniform(25, 100, size=(1,)).astype('int0')[0]

    color = np.random.uniform(0, 255, size=3).astype('int0')
    color = tuple(map(int, color))
    cv2.circle(image, (x, y), r, color, -1)
    cv2.imwrite('{}/{}.jpg'.format(args['output'], uuid.uuid4()), image)

# generate a single rectangle
image = np.zeros((500, 500, 3), dtype='uint8')
topleft = np.random.uniform(25, 225, size=(2,)).astype('int0')
botRight = np.random.uniform(250, 400, size=(2,)).astype('int0')
color = np.random.uniform(0, 255, size=3).astype('int0')
color = tuple(map(int, color))
cv2.rectangle(image, tuple(topleft), tuple(botRight), color, -1)
cv2.imwrite('{}/{}.jpg'.format(args['output'], uuid.uuid4()), image)

