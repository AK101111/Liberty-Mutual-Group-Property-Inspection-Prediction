import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import csv
import sys
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats



# Load the dataset
data=pd.read_csv('train_preprocessed.csv', sep=',',header=None)
#data=data.values
data = data.as_matrix()


# Use only one feature
#data_T1_V1 = data[:, 2]
#data_T1_V1 = np.array([data_T1_V1]).T

total_data = data[:,2:]
total_data = np.arrray([total_data]).T

hazard = data[:,1]
hazard = np.array([hazard]).T


# Split the data into training/testing sets

#data_T1_V1_train = data_T1_V1[:40000]
#data_T1_V1_test = data_T1_V1[40000:]

total_data_train = total_data[:40000]
total_data_test = total_data[40000:] 


# Split the targets into training/testing sets
data_hazard_train = hazard[:40000]
data_hazard_test = hazard[40000:]

#create PCA object
pca = PCA(n_components=100) #n_components

# Train the model using the training sets
pca.fit(total_data_train)

# The pca_score	/ explained in 'http://www.miketipping.com/papers/met-mppca.pdf' /
pca_score = pca.explained_variance_ratio_

# The number of components
V = pca.components_

# Contains the new basis vector instead of the standard e(i,j,k,...)
pca_axis = V.T * pca_score / pca_score.min()


print('PCA_SCORE: \n', pca_score)
print('PCA_number_components: \n', V)