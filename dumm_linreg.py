# Done with Dummy encoding and linreg
# Score = 0.3451
import numpy as np
import csv
from sklearn import linear_model, datasets, preprocessing
from scipy import stats
from sklearn.decomposition import PCA
import pandas as pd
from gini_score import normalized_gini

#train = pd.read_csv('train.csv')

#columns = train.columns

#train = np.array(train)	
	
with open('train_preprocessed.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter = ',')
	x = list(reader)
	result = np.array(x)
   
#result_X = train

#print result[2, :]
#result = result[:, [1, 2, 3, 4]];
result_X = result.astype('float')
#result_X = preprocessing.scale(result_X)

X_feats = result_X[:, 2:]
X_target = result_X[:, 1]

#pca = PCA(100)
#X_feats = pca.fit_transform(X_feats)

############clf = lin reg###############
clf = linear_model.LinearRegression()
clf.fit(X_feats, X_target)

X_test = X_feats[:, :]
Y_test = result_X[:, 1]
########################################
############ test data #################

#with open('test.csv') as csvfile:
#	reader = csv.reader(csvfile, delimiter = ',')
#	x = list(reader)
#	test_dat = np.array(x)

#test_dat = test_dat[2:, 2:]
#test_dat_n = test_dat.astype('float')

#X_test = test_dat_n

# Predicted on the train set itself
predicted_x = clf.predict(X_test)
a = gini(predicted_x)
print a



#print clf.predict(X_test)
#print Y_test
#print clf.score(X_feats[100:, :], result_X[100:, 1])
#print X_feats[:, 2]

#np.savetxt('clfcoeff2.txt', clf.coef_)
#result_X = np.column_stack(result + [[1]*len(result[0])])
#beta_hat = np.linalg.lstsq(result[1:[1, 2, 3]], result[1:,[0]].T)[0]
#print clf.coef_
