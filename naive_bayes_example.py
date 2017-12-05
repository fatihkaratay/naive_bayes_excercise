import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from time import time

# Input training data
training_points = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
training_labels = [1, 1, 1, 2, 2, 2]
X = np.array(training_points)
Y = np.array(training_labels)

# Create Naive Bayes classifier
clf = GaussianNB()
training_time = time()
clf.fit(X, Y)
print "training time:", round(time()-training_time, 3), "s"

# Classify test data with the classifier
test_points = [[1, 1], [2, 2], [3, 3], [4, 3]]
test_labels = [2, 2, 2, 1]
predict_time = time()
predicts = clf.predict(test_points)
print "prediction time:", round(time()-predict_time, 3), "s"
print predicts

print "Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(test_labels, predicts)
