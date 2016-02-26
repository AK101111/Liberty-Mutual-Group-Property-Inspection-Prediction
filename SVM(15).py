#Score = 0.209 # SVM Classifier
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import csv
from sklearn import linear_model, datasets, preprocessing, svm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from gini_score import normalized_gini

with open('train_one_hot.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter = ',')
	x = list(reader)
	result = np.array(x)

result_X = result.astype('float')

X_feats = result_X[:, 2:]
X_target = result_X[:, 1]

for i in range(1, len(X_target)):
	if (X_target[i] > 15):
		X_target[i] = 15

############clf = svm c ###############
clf = svm.SVC()
clf.fit(X_feats, X_target)

X_test = X_feats
Y_test = X_target
############

predicted_x = clf.predict(X_test)

a = normalized_gini(predicted_x, Y_test)
print a 	#("GINI Score : ", a)