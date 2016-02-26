import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
import csv
from sklearn import linear_model, datasets, preprocessing
from scipy import stats

##open file with categorical variables encoded in one hot encoding
with open('train_one_hot.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter = ',')
	x = list(reader)
	result = np.array(x)

result = result.astype('float')		#convert int data to float
result_X = result[:, 2:]		#training data
result_target = result[:, 1]		#target values
#result_X = preprocessing.scale(result)

####### Linear reg fitting
clf = linear_model.LinearRegression()
clf.fit(result_X, result_target)

np.savetxt('clfcoeff2.txt', clf.coef_)
