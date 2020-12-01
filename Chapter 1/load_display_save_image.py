# USAGE
# python load_display_save-image.py --image florida_trip.png --output florida
# python load_display_save_image.py --image grand_canyon.png --output grand


# import packages
import argparse
import cv2

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i',  '--image', required=True, help='path to the image')
ap.add_argument('-o', '--output', required=True, help='path to new image')
args = vars(ap.parse_args())


# load image and show basic information
image = cv2.imread(args['image'])
print('width: {} pixel'.format(image.shape[1]))
print('height: {} pixel'.format(image.shape[0]))
print('channels: {} pixel'.format(image.shape[2]))

# show the image and wait for a keypress
cv2.imshow('Image', image)
cv2.waitKey(0)

# save and convert filetypes
cv2.imwrite('{}.jpg'.format(args['output']), image)
