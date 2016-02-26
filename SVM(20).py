import pandas as pd
import numpy as np 
from sklearn import preprocessing, svm, linear_model
import xgboost as xgb
from sklearn.svm import SVR

#load train and test 
train  = pd.read_csv('train.csv', index_col=0)
test  = pd.read_csv('test.csv', index_col=0)

labels = train.Hazard

columns = train.columns
test_ind = test.index

Categorical_vars_new = ['T1_V4','T1_V5','T1_V6','T1_V7','T1_V8','T1_V9','T1_V11','T1_V12','T1_V15','T1_V16','T1_V17','T2_V3','T2_V5','T2_V11','T2_V12','T2_V13']

############## dummy encode ###################

ztest = []
ztrain = []
for x in train:	#Categorical_vars_old:
	if x in Categorical_vars_new:
		y = pd.get_dummies(train[x])
		for j in y:
			ztrain.append(y[j].tolist())
	else:
		ztrain.append(train[x].tolist())

for x in test:	#Categorical_vars_old:
	if x in Categorical_vars_new:
		y = pd.get_dummies(test[x])
		for j in y:
			ztest.append(y[j].tolist())
	else:
		ztest.append(test[x].tolist())

ztrain = np.array(ztrain)
ztest = np.array(ztest)
ztrain = ztrain.transpose()
ztest = ztest.transpose()

ztrain = ztrain.astype(float)
ztest = ztest.astype(float)

X_feats = ztrain[:, 1:]
X_target = labels

X_feats_test = ztest

X_target = np.array(X_target)
X_target = X_target.astype('float')
for i in range(1, len(X_target)):
	if (X_target[i] > 20):
		X_target[i] = 20

############ clf = svc() ###############
clf = svm.SVC(C = 5, gamma = 5)
clf.fit(X_feats, X_target)

X_test = X_feats
Y_test = X_target
#######################################

predecs = clf.predict(X_feats_test)

#generate solution
predecs = pd.DataFrame({"Id": test_ind, "Hazard": predecs})
predecs = predecs.set_index('Id')
predecs.to_csv('svm_20_benchmark.csv')                