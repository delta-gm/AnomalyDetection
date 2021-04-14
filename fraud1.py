import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import statistics
import scipy.stats
import pycaret as pc 

#
# # Importing libraries
# !pip install pycaret
# !pip install boruta
# from tkinter.filedialog import askopenfilename
# import pandas as pd
# import numpy as np
# import random
# import sklearn
# import re
# from sklearn.ensemble import RandomForestClassifier
# from boruta import BorutaPy
# from sklearn.metrics import roc_curve,roc_auc_score,f1_score,cohen_kappa_score,precision_score,recall_score,confusion_matrix
# from sklearn.model_selection import ParameterGrid, ParameterSampler
# from time import time
# from sklearn.metrics import classification_report,accuracy_score
# from sklearn.ensemble import IsolationForest
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.svm import OneClassSVM
# from pylab import rcParams
# from scipy.stats import itemfreq
# from pycaret.anomaly import *
# from sklearn.metrics import roc_curve
# import matplotlib.pyplot as plt
#


# from https://towardsdatascience.com/anomaly-detection-made-simple-70775c914377
# Use isolation forest, angle-base outlier detection, subspace outlier detection, and minimum covariance determinant
creditcard = pd.read_csv('creditcard.csv')

Fraud = creditcard[creditcard['Class']==1]
Nofraud = creditcard[creditcard['Class']==0]
outlier_fraction = len(Fraud)/float(len(Nofraud))

#checking if the outlier fraction is almost same to the original dataset
Fraud = data[data['Class']==1]
Nofraud = data[data['Class']==0]
outlier_fraction = len(Fraud)/float(len(Nofraud))

data = creditcard.sample(frac=0.1,random_state=1).reset_index(drop=True)

X = setup(data, ignore_features = ['Class'], session_id = 123)
Y = data['Class']

# Isolation Forest 
#  Isolation forest algorithm segregates observations by randomly selecting a feature and then randomly selecting a split 
# value between the maximum and minimum values of selected feature similarly constructing the separation by creating random 
# decision trees⁶. Thus, an anomaly score is calculated as the number of conditions required to separate a given observation.

iforest = create_model('iforest', fraction = outlier_fraction)
print(iforest)

iforest_results = assign_model(iforest)
iforest_results[['Score','Label']].head()

n_errors = (iforest_results['Label'] != iforest_results['Class']).sum()
print("Isolation forest parameters n_estimators:{}".format(iforest))
print("number of errors: {}".format(n_errors))
print("Accuracy Score :")
print(accuracy_score(iforest_results['Class'],iforest_results['Label'] ))
print("Classification Report :")
print(classification_report(iforest_results['Class'],iforest_results['Label']))
print("confusion matrix:")
print(confusion_matrix(iforest_results['Class'],iforest_results['Label']))
print("ROC_AUC:")
print(roc_auc_score(iforest_results['Class'],iforest_results['Score']))

fpr, tpr, thresholds = roc_curve(iforest_results['Class'],iforest_results['Score'])
plt.plot(fpr, tpr, 'k-', lw=2)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# ANGLE-BASE OUTLIER DETECTION

abod = create_model('abod',fraction=outlier_fraction,verbose=True)
print(abod)

abod_results = assign_model(abod)
abod_results[['Score','Label']].head()

# Run Classification Metrics
n_errors = (abod_results['Label'] != abod_results['Class']).sum()
print("Angle-base Outlier Detection parameters n_estimators:{}".format(abod))
print("number of errors: {}".format(n_errors))
print("Accuracy Score :")
print(accuracy_score(abod_results['Class'],abod_results['Label'] ))
print("Classification Report :")
print(classification_report(abod_results['Class'],abod_results['Label']))
print("confusion matrix:")
print(confusion_matrix(abod_results['Class'],abod_results['Label']))
print("ROC_AUC:")
print(roc_auc_score(abod_results['Class'],abod_results['Score']))

fpr, tpr, thresholds = roc_curve(abod_results['Class'],abod_results['Score'])
plt.plot(fpr, tpr, 'k-', lw=2)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# SUBSPACE OUTLIER DETECTION

sod = create_model('sod',fraction=outlier_fraction)
print(sod)

sod_results = assign_model(sod)
sod_results[['Score','Label']].head()

# Run Classification Metrics
n_errors = (sod_results['Label'] != sod_results['Class']).sum()
print("Subspace Outlier Detection parameters n_estimators:{}".format(sod))
print("number of errors: {}".format(n_errors))
print("Accuracy Score :")
print(accuracy_score(sod_results['Class'],sod_results['Label'] ))
print("Classification Report :")
print(classification_report(sod_results['Class'],sod_results['Label']))
print("confusion matrix:")
print(confusion_matrix(sod_results['Class'],sod_results['Label']))
print("ROC_AUC:")
print(roc_auc_score(sod_results['Class'],sod_results['Score']))

fpr, tpr, thresholds = roc_curve(sod_results['Class'],sod_results['Score'])
plt.plot(fpr, tpr, 'k-', lw=2)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# MCD

mcd = create_model('mcd', fraction = outlier_fraction)
print(mcd)

mcd_results = assign_model(mcd)
mcd_results[['Score','Label']].head()

# Run Classification Metrics
n_errors = (mcd_results['Label'] != mcd_results['Class']).sum()
print("Subspace Outlier Detection parameters n_estimators:{}".format(mcd))
print("number of errors: {}".format(n_errors))
print("Accuracy Score :")
print(accuracy_score(mcd_results['Class'],mcd_results['Label'] ))
print("Classification Report :")
print(classification_report(mcd_results['Class'],mcd_results['Label']))
print("confusion matrix:")
print(confusion_matrix(mcd_results['Class'],mcd_results['Label']))
print("ROC_AUC:")
print(roc_auc_score(mcd_results['Class'],mcd_results['Score']))

fpr, tpr, thresholds = roc_curve(mcd_results['Class'],mcd_results['Score'])
plt.plot(fpr, tpr, 'k-', lw=2)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()
# get ROC_AUC 