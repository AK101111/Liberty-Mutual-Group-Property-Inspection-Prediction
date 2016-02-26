#sample3.py Simulate SVR Score = 0.4345
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import csv
from sklearn import linear_model, datasets, preprocessing
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from gini_score import normalized_gini
from sklearn import tree

with open('train_preprocessed.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter = ',')
	x = list(reader)
	result = np.array(x)

#result = result[:, [1, 2, 3, 4]];
result_X = result.astype('float')
#result_X = preprocessing.scale(result_X)

X_feats = result_X[:, 2:]
X_target = result_X[:, 1]

#pca = PCA(100)
#X_feats = pca.fit_transform(X_feats)

############clf = regression tree###############
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_feats, X_target)

X_test = X_feats
Y_test = X_target
############

predicted_x = clf.predict(X_test)

a = normalized_gini(predicted_x, Y_test)
print a#("GINI Score : ", a)

P = clf.get_params()
#np.savetxt('svr1.txt', clf.coef_)
#result_X = np.column_stack(result + [[1]*len(result[0])])
#beta_hat = np.linalg.lstsq(result[1:[1, 2, 3]], result[1:,[0]].T)[0]
#print clf.coef_

#from sklearn.externals.six import StringIO  
#import pydot 
#dot_data = StringIO() 
#tree.export_graphviz(clf, out_file=dot_data) 
#graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
#graph.write_pdf("decision.pdf") 

