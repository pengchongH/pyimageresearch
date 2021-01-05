from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import datasets
from skimage import exposure
import numpy as np
import imutils
import cv2
import sklearn

# load the MINST digits dataset
mnist = datasets.load_digits()

# split the dataset to train, validation and test set:0.65:0.1:0.25
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data), mnist.target, test_size=0.25, random_state=42)
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1, random_state=84)

# show the size of each dataset
print('training data points: {}'.format(len(trainLabels)))
print('validation data points: {}'.format(len(valLabels)))
print('testing data points: {}'.format(len(testLabels)))

# initialize the values of k for knn classifier along with the list of accuracies for each value of k
kVals = range(1, 30, 2)  # ensure k is an odd number
accuracies = []

for k in kVals:
    # train
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabels)

    # evaluate
    score = model.score(valData, valLabels)
    print('k={}, accuracy={:.2f}%'.format(k, score*100))
    accuracies.append(score)

# find the value of k that has the largest accuracy
i = int(np.argmax(accuracies))
print('k={} achieved highest accuracy of {:.2f}% on validation data.'.format(kVals[i], accuracies[i]*100))

# re_train by using the best k
model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(trainData, trainLabels)
predictions = model.predict(testData)

# show a classification report
print('EVALUATION ON TESTING DATA')
print(classification_report(testLabels, predictions))

# examining some examples
for i in list(map(int, np.random.randint(0, high=len(testLabels), size=(5,)))):
    image = testData[i]
    prediction = model.predict(image.reshape(1, -1))[0]

    image = image.reshape((8, 8)).astype('uint8')
    image = exposure.rescale_intensity(image, out_range=(0, 255)).astype('uint8')
    image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

    print('I think that digit is: {}'.format(prediction))
    cv2.imshow('Image', image)
    cv2.waitKey(0)

