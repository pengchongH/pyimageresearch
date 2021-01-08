from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import sklearn

# generate the XOR data
tl = np.random.uniform(size=(100, 2)) + np.array([-2.0, 2.0])
tr = np.random.uniform(size=(100, 2)) + np.array([2.0, 2.0])
bl = np.random.uniform(size=(100, 2)) + np.array([-2.0, -2.0])
br = np.random.uniform(size=(100, 2)) + np.array([2.0, -2.0])
X = np.vstack([tl, tr, br, bl])
Y = np.hstack([[1] * len(tl), [-1] * len(tr), [1] * len(br), [-1] * len(bl)])

# split dataset
(trainData, testData, trainLabels, testLabels) = train_test_split(X, Y, test_size=0.25, random_state=42)

# train the linear SVM model
print('[RESULT] SVM w/ Linear Kernel')
model = SVC(kernel='linear')
model.fit(trainData, trainLabels)
print(classification_report(testLabels, model.predict(testData)))

print('[RESULT] SVM w/ Polynomial Kernel')
model = SVC(kernel='poly', degree=2, coef0=1)
model.fit(trainData, trainLabels)
print(classification_report(testLabels, model.predict(testData)))
