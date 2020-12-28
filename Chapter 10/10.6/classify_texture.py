# Haralick texture

from sklearn.svm import LinearSVC
import argparse
import mahotas
import glob
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--training', required=True, help='path to the dataset of textures')
ap.add_argument('-t', '--test', required=True, help='path to the test images')
args = vars(ap.parse_args())

print('[INFO] extracting features...')
data = []
labels = []

for imagePath in glob.glob(args['training'] + '/*.png'):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture = imagePath[imagePath.rfind('/') + 1:].split('_')[0]

    # extract Haralick texture features in 4 directions, then take the mean of each direction
    features = mahotas.features.haralick(gray).mean(axis=0)

    data.append(features)
    labels.append(texture)

print('[INFO] training model...')
model = LinearSVC(C=10.0, random_state=42)
model.fit(data, labels)


print('[INFO] classifying...')
for imagePath in glob.glob(args['test'] + '/*.png'):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = mahotas.features.haralick(gray).mean(axis=0)

    # classify the test image
    pred = model.predict(features.reshape(1, -1))[0]
    cv2.putText(image, pred, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    cv2.imshow('Image', image)
    cv2.waitKey(0)



