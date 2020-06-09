#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

clf = SVC(kernel='rbf', C=10000.0)
t0 = time()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
print("Tempo de treinamento: ", round(time()-t0,3), "s")
accuracy = accuracy_score(labels_test, pred)


# prediction_10 = clf.predict(features_test[10])
# prediction_26 = clf.predict(features_test[26])
# prediction_50 = clf.predict(features_test[50])

# print(prediction_10)
# print(prediction_26)
# print(prediction_50)

print(sum(pred))

print(accuracy)

#########################################################


