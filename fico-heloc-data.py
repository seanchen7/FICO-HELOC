# coding: utf-8

import os
import numpy as np
import pandas as pd
# import sklearn as sk
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

# import matplotlib.pyplot as plt
import pydotplus
from IPython.display import Image
import seaborn as sns


# User-defined functions
def model_summary(y_test, tc_predict, dtc_cv_score):
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, dtc_predict))
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(y_test, dtc_predict))
    print('\n')
    print("=== All AUC Scores ===")
    print(dtc_cv_score)
    print('\n')
    print("=== Mean AUC Score ===")
    print("Mean AUC Score - Random Forest: ", dtc_cv_score.mean())


#%% Import and prepare data

os.chdir("/Users/sean/Documents/GitHub/FICO-Challenge")

# FICO Home Equity Line of Credit (HELOC) Dataset (https://community.fico.com/s/explainable-machine-learning-challenge)
heloc = pd.read_csv('heloc_dataset_v1.csv')

# The target variable to predict is a binary variable called RiskPerformance. 
target_names = ['Good', 'Bad']
heloc.loc[heloc['RiskPerformance']=='Good','Class'] = target_names[0]
heloc.loc[heloc['RiskPerformance']=='Bad','Class'] = target_names[1]
heloc['Class'] = heloc['Class'].astype('category')


#Explore data
dt1 = heloc.describe()
heloc.dtypes
heloc.groupby('Class')['Class'].count()

sns.boxplot(x="MaxDelqEver", y="Class", data=heloc)


#%% Define predcitors and target variable
X = heloc.drop(columns=['Class', 'RiskPerformance'])
y = heloc['Class']
feature_names = X.columns


# Implement train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)


#%% Create a simple tree model
dtc = tree.DecisionTreeClassifier(max_depth=5)
dtc = dtc.fit(X_train, y_train)
dtc_predict = dtc.predict(X_test) # predictions
dtc_cv_score = cross_val_score(dtc, X, y, cv=10, scoring='roc_auc') #scores

# Model summary
model_summary(y_test, dtc_predict, dtc_cv_score)

#tree.plot_tree(dtc)
dot_data = tree.export_graphviz(dtc, feature_names=feature_names, class_names=target_names, 
                                out_file=None, filled = True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph

#%% Create a Random Forest Model

rfc = RandomForestClassifier()
rfc.fit(X_train,y_train) #fit
rfc_predict = rfc.predict(X_test) # predictions
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc') #scores

# Model summary
model_summary(y_test, rfc_predict, rfc_cv_score)


# Tuning Hyperparameters

# Optimize select hyperparamaters
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)] # number of trees in random forest
max_features = ['auto', 'sqrt'] # number of features at every split
max_depth = [int(x) for x in np.linspace(100, 500, num = 11)] # max depth
max_depth.append(None)

# Create random grid
random_grid = {
 'n_estimators': n_estimators,
 'max_features': max_features,
 'max_depth': max_depth
 }

# Random search of parameters
rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, 
                                cv = 3, verbose=2, random_state=42, n_jobs = -1)
rfc_random.fit(X_train, y_train) # Fit the model
print(rfc_random.best_params_) # Print results


# Final Model
# Refit the model with tuned parameters
rfc = RandomForestClassifier(n_estimators=500, max_depth=100, max_features='sqrt')
rfc.fit(X_train,y_train) #fit
rfc_predict = rfc.predict(X_test) # predictions
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc') #scores

# Model summary
model_summary(y_test, rfc_predict, rfc_cv_score)


# Feature Importance

importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(X.columns, rfc.feature_importances_):
    feats[feature] = importance 
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})

importances.sort_values(by='Gini-importance').plot(kind='bar', rot=90) #Plot
importances.head() # Print the feature ranking

