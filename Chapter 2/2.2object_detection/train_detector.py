from imutils import paths
from scipy.io import loadmat
from skimage import io
import argparse
import dlib
import sys

# handle Python3 compatibility
if sys.version_info > (3,):
    long = int

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--class', required=True, help='Path to the CALTECH-101 class images')
ap.add_argument('-a', '--annotations', required=True, help='Path to the CALTECH-101 class annotations')
ap.add_argument('-o', '--output', required=True, help='Path to the output detector')
args = vars(ap.parse_args())


# default training options: HOG + linear SVM
print('[INFO] gathering images and bounding boxes...')
options = dlib.simple_object_detector_training_options()
images = []
boxes = []

for imagePath in paths.list_images(args['class']):
    # extract the ImageID and annotation
    imageID = imagePath[imagePath.rfind('/') + 1:].split('_')[1]
    imageID = imageID.replace('.jpg', '')
    p = '{}/annotation_{}.mat'.format(args['annotations'], imageID)
    annotations = loadmat(p)['box_coord']

    bb = [dlib.rectangle(left=long(x), top=long(y), right=long(w), bottom=long(h)) for (y, h, x, w) in annotations]
    boxes.append(bb)

    images.append(io.imread(imagePath))

# train the object detector
print('[INFO] training detector...')
detector = dlib.train_simple_object_detector(images, boxes, options)

# dump the classfier to file
print('[INFO] dumoing classfier to file...')
detector.save(args['output'])

# visualize the results of the detector
win = dlib.image_window()
win.set_image(detector)
dlib.hit_enter_to_continue()
