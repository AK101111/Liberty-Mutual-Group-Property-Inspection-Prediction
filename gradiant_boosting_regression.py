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
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error


###############################################################################
# Load data
with open('train_preprocessed.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter = ',')
	x = list(reader)
	result = np.array(x)

#result = result[:, [1, 2, 3, 4]];
result_X = result.astype('float')
#result_X = preprocessing.scale(result_X)

X_feats = result_X[:, 2:][:40000]
X_target = result_X[:, 1][:40000]

X_test = result_X[:, 2:][40000:]
Y_test = result_X[:, 1][40000:]


###############################################################################
# Fit regression model
params = {'n_estimators': 300, 'max_depth': 9, 'min_samples_split': 1,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_feats, X_target)
mse = mean_squared_error(Y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)

predicted_x = clf.predict(X_test)

a = normalized_gini(predicted_x, Y_test)
print a#("GINI Score : ", a)

P = clf.get_params()

with open('grad_boost_reg.csv', 'wb') as csvfile:
    swriter = csv.writer(csvfile, delimiter=',')
    swriter.writerow([x for x in P])

###############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, boston.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
