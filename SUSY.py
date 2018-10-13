# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
#import matplotlib as plt
#from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('SUSY.csv', header=None, nrows = 150000).values

data_train, data_test, label_train, label_test = train_test_split(data[:,1:], data[:,0], test_size = 0.8, stratify = data[:,0])

#clf = DecisionTreeClassifier(random_state=0)

clf = SVC()

model = clf.fit(data_train, label_train)
label_predict = model.predict(data_test)
accuracy = np.mean(label_test == label_predict)
print(accuracy)

#score = cross_val_score(clf, data_train, label_train, cv = 10, verbose = 3)
#print(score)

#print(data)
