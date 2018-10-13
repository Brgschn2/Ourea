# -*- coding: utf-8 -*-
"""
2018.10.13
Matt

First project implementing basic ML classification
using SUSY dataset (monte carlo simulated)
https://archive.ics.uci.edu/ml/datasets/SUSY

SciKit Learn resources used
Educated by Anil and Orlando
"""

import numpy as np
import pandas as pd
#import matplotlib as plt
#from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# SUSY.csv has 5,000,000 events; <10% is best for following read:
data = pd.read_csv('SUSY.csv', header=None, nrows = 100000).values

# Using train_test_split, separates data into 80% for training, 20% for testing
# Stratisfying will shuffle the data following signal/noise basis??
data_train, data_test, label_train, label_test = train_test_split(data[:,1:],
    data[:,0], test_size = 0.8, stratify = data[:,0])

# Establishing clf using Decision Tree Classification
#clf = DecisionTreeClassifier(random_state=0)

# Establishing clf using sklearn.svm.SVC (specify gamma = 'auto' in preparation for update)
clf = SVC(gamma = 'auto')

# Manual fit training, with prediction and accuracy averaging
# (Rather than using cross_val_score)
model = clf.fit(data_train, label_train)
label_predict = model.predict(data_test)
accuracy = np.mean(label_test == label_predict)
print(accuracy)


# An option for establishing cross validation score using the chosen clf
# and data as given (comparing data and label to train for)
# verbosity 3 should give each of 10 cv instances time to complete and score (and some other garbage)

#score = cross_val_score(clf, data_train, label_train, cv = 10, verbose = 3)
#print(score)

#print(data)
