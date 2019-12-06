import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier as ada
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, recall_score, classification_report
from sklearn.metrics import auc
import os
import sys
dataFile = os.getcwd()+'/data/creditcard.csv'
allData = pd.read_csv(dataFile)
#DEFAULTS
arg = sys.argv[1]
feature_cutoff = int(arg)
print("Opening file at " + os.getcwd() + '\\results\\acc_vals\\feat_cutoff_' + str(arg) + '.txt')
print("Feature Cutoff set at " + str(arg))
# result_file = open(os.getcwd() + '/results/acc_vals/feat_cutoff_' + str(arg) + '.txt', 'w+')
# result_file.write("Cut " + str(feature_cutoff) + " features out of 29")
test_split = 0.3
weight = 10
print("Test split set at " + str(test_split))
print("Weight set at ", weight)
# END_DEFAULTS

print("Starting...")
numeric_features = allData.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print(corr['Class'].sort_values(ascending=False))
corr_list = corr['Class'].sort_values(ascending=True)[(int(len(corr)/2)) - feature_cutoff:(int(len(corr)/2)) + feature_cutoff]

# result_file.write("\nCut features are:\n" + str(corr_list.axes[0].tolist()))
axes_list = corr_list.axes[0].tolist()

allData = allData.drop(axes_list, axis=1)

allData_X = allData[allData.axes[1].tolist()]
allData_X = allData_X.drop('Class', axis=1)
print(axes_list)
print(allData_X.axes[1].tolist())
# result_file.write("\nUsing features:\n" + str(allData_X.axes[1].tolist()))
# result_file.write("\n\nTrain Split: " + str(1 - test_split))
# result_file.write("\nTest Split: " + str(test_split))
allData_y = allData['Class']

train_X, test_X, train_y, test_y = train_test_split(allData_X, allData_y, test_size=test_split, random_state=0)

weight_list = (train_y*weight + 1)

lr = LogisticRegression(penalty='l1')
ab_clf = ada(DecisionTreeClassifier(max_depth=1), n_estimators=50)
rf_clf = rfc(n_estimators=50)

print("Starting Logistic Regression")
lr.fit(train_X, train_y, sample_weight=weight_list)
predictions = lr.predict(test_X)

lr_recall_score = recall_score(test_y, predictions)
lr_report = classification_report(test_y, predictions)

lr_f1 = f1_score(test_y, predictions)
# result_file.write("\n\nLogistic Regression:")
print('\n' + lr_report)
# result_file.write("\n" + lr_report)
print("Finished Logistic Regression")

print("Starting Adaboost")
ab_clf.fit(train_X, train_y, sample_weight=weight_list)
predictions = ab_clf.predict(test_X)

ab_recall_score = recall_score(test_y, predictions)
ab_report = classification_report(test_y, predictions)
ab_f1 = f1_score(test_y, predictions)

# result_file.write("\n\nAdaboost:")
print('\n' + ab_report)
# result_file.write('\n' + ab_report)

print("Finished Adaboost")

print("Starting Random Forest")
rf_clf.fit(train_X, train_y, sample_weight=weight_list)
predictions = rf_clf.predict(test_X)

rf_recall_score = recall_score(test_y, predictions)
rf_report = classification_report(test_y, predictions)
rf_f1 = f1_score(test_y, predictions)

# result_file.write("\n\nRandom Forest: ")
print('\n' + rf_report)
# result_file.write('\n' + rf_report)
print("Finished Random Forest")
print("Closing " + os.getcwd() + '\\results\\feat_cutoff_' + str(arg) + '.txt')
# result_file.close()
print("Closed")
