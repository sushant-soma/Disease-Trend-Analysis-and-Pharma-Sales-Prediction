# gradient boosting for regression in scikit-learn
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from matplotlib import pyplot
import pandas as pd
import csv
import time

# Importing the dataset
datas = pd.read_csv('all_year.csv')
# define dataset

X, y = make_regression(n_samples=10, n_features=1)

print(y)
# evaluate the model
model = GradientBoostingRegressor()
cv = RepeatedKFold(n_splits=2)
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model = GradientBoostingRegressor()
model.fit(X, y)
#yh=model.predict()
# make a single prediction
# row = [[2.02220122, 0.31563495, 0.82797464, -0.30620401, 0.16003707, -1.44411381, 0.87616892, -0.50446586, 0.23009474, 0.76201118]]
# yhat = model.predict(row)
# print('Prediction: %.3f' % yhat[0])