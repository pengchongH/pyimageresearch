from pyimagesearch.face_recognition.datasets import load_caltech_faces
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to CALTECH Faces dataset')
ap.add_argument('-s', '--sample_size', type=int, default=10, help='# of example samples')
args = vars(ap.parse_args())

# load the CALTECH faces dataset
print('[INFO] loading CALTECH Faces dataset...')
(training, testing, names) = load_caltech_faces(args['dataset'], min_faces=21, test_size=0.25)

# encode the labels, transforming them from strings into integers since OpenCV does not like string as training data
le = LabelEncoder()
le.fit_transform(training.target)

# creating the LBP face recognition
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)

print('[INFO] training face recognizer...')
recognizer.train(training.data, le.transform(training.target))

# initialize the list of predictions and confidence scores
print('[INFO] gathering predictions...')
predictions = []
confidence = []

for i in range(0, len(testing.data)):
    print('{} of {}'.format(str(i), str(len(testing.data))))
    # classify the face and update the list of the predictions and confidence scores
    (prediction, conf) = recognizer.predict(testing.data[i])
    predictions.append(prediction)
    confidence.append(conf)

# show the classification report
print(classification_report(le.transform(testing.target), predictions, target_names=np.unique(names)))

# loop over the the desired number of samples
for i in np.random.randint(0, high=len(testing.data), size=(args["sample_size"],)):
	# resize the face to make it more visable, then display the face and the prediction
	print("[INFO] Prediction: {}, Actual: {}, Confidence: {:.2f}".format(
		le.inverse_transform([predictions[i]])[0], testing.target[i], confidence[i]))
	face = testing.data[i]
	face = imutils.resize(face, width=face.shape[1] * 2, inter=cv2.INTER_CUBIC)
	cv2.imshow("Face", face)
	cv2.waitKey(0)

