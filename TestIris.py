# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 23:42:33 2018

@author: Matthew
"""

# Check the versions of libraries

# Load libraries
import pandas
#print('pandas: {}'.format(pandas.__version__))
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# shape
#print(dataset.shape)

# head
#print(dataset.head(20))

# descriptions
#print(dataset.describe())