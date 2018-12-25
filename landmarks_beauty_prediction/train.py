from sklearn.svm import SVR
from sklearn import linear_model,neighbors
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import decomposition
from sklearn.ensemble import RandomForestRegressor
import statistics
import scipy.stats
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

def feature_importances(kernel,X,y):

    rfecv = RFECV(estimator=kernel, step=1,
                  scoring='neg_mean_absolute_error')
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

def corr_score(kernel, X_train,X_test,y_train,y_test):
    kernel.fit(X_train, y_train)
    y_pred = kernel.predict(X_test)
    temp = np.corrcoef(y_pred, y_test)
    corr = temp[0, 1]
    return corr

#read features
with open('features.txt') as f:
    features = f.readlines()
features = [x.strip() for x in features]

# read labels
with open('labels.txt') as f:
    labels = f.readlines()
labels = [x.strip() for x in labels]

X =[]
y = []
for item in features:
    mylist = item.split(',')
    X.append(list(map(float, mylist)))

for item in labels:
    y.append(float(item))

# define the regressors 
svr_lin = SVR(kernel='linear', C=1e3)

pca = decomposition.PCA(n_components=100)
pca.fit(X)
X_train = pca.transform(X)

scaler = MinMaxScaler()
X_transform = scaler.fit_transform(X_train) # significantly reduces the training time

# calculate mean squared error scores for each regressor
MSE_linear = cross_val_score(svr_lin,X_transform,y,cv =10, scoring = 'neg_mean_squared_error')

# RMSE Scores
print("Root Mean squared error for linear SVR: %.2f" % math.sqrt((MSE_linear.mean())))


