import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import linear_model, datasets, preprocessing
from scipy import stats

#Read from the encoded file
with open('train_one_hot.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter = ',')
	x = list(reader)
	result = np.array(x)

result = result.astype('float')			#Convert to float
result_scaled = preprocessing.scale(result)	#Scaling of data
means = result.mean(axis = 0) 			#Calculate the means vertically
co = 0
var = []

#Variance
for ii in range(0, result.shape[1]):
	co = np.std(zip(*result)[ii])
	var.append(co)

np.savetxt('tr_one_hot_mean.txt', means)	#Write Means
np.savetxt('tr_one_hot_var.txt', var)		#Write Variances
