# coding: utf-8

import os
import numpy as np
import pandas as pd
# Scikit-learn package for machine learning models
import sklearn as sk
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
# Packages for graphics
import pydotplus
from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Change directory
file_path = '/Users/sean/Documents/GitHub/FICO-Challenge'
os.chdir(file_path)
# os. getcwd() 

#%% Import and prepare data

# FICO Home Equity Line of Credit (HELOC) Dataset (https://community.fico.com/s/explainable-machine-learning-challenge)
df = pd.read_csv(file_path + '/Data/heloc_dataset_v1.csv')
df.dtypes

# The target variable to predict is a binary variable called RiskPerformance. 
target_variable = 'RiskPerformance'
df[target_variable].drop_duplicates()
target_names = ['Good', 'Bad']

# Create standardized target varibale column
df.loc[df[target_variable]==target_names[0],'Class'] = 0
df.loc[df[target_variable]==target_names[1],'Class'] = 1
df['Class'] = df['Class'].astype(int)
df.groupby('Class')['Class'].count() # Count of categorical variable

# Convert categorical variables to proper types
var_category = [target_variable]  #Keep 'Class' as numerical since it only has two values
df[var_category] = df[var_category].astype('category')

# Feature variables
features = list(df.columns)
features.remove(target_variable)
features.remove('Class')

#%% Data cleaning 

# Data cleaning and exclusion
## Special values according to data dictionary
df[df==-9].count() # -9 No Bureau Record or No Investigation

dt1 = df[df==-8].count() # -8 No Usable/Valid Trades or Inquiries
dt1[dt1>0]
#!!! Need to figure out strategy
# MSinceOldestTradeOpen                  239
# MSinceMostRecentDelq                   176
# MSinceMostRecentInqexcl7days           476
# NetFractionRevolvingBurden             186
# NetFractionInstallBurden              3419
# NumRevolvingTradesWBalance             156
# NumInstallTradesWBalance               861
# NumBank2NatlTradesWHighUtilization     583
# PercentTradesWBalance                   18

df[df==-7].count() # -7 Condition not Met (e.g. No Inquiries, No Delinquencies)
#MSinceMostRecentDelq, MSinceMostRecentInqexcl7days
#!!! Should we use an artificially large number to relace -7 given the relationship constraint with the target variable

## MaxDelq2PublicRecLast12M	
df.groupby('MaxDelq2PublicRecLast12M')['MaxDelq2PublicRecLast12M'].count() 
# No adjustments needed - Values 0-7 are monotonically decreasing
# value      meaning                                                                       	
# 0	derogatory comment
# 1	120+ days delinquent
# 2	90 days delinquent
# 3	60 days delinquent
# 4	30 days delinquent
# 5, 6	unknown delinquency
# 7	current and never delinquent
# 8, 9	all other

## MaxDelqEver	
df.groupby('MaxDelqEver')['MaxDelqEver'].count() 
# No adjustments needed - Values 2-8 are monotonically decreasing
# value     meaning	
# 1	No such value
# 2	derogatory comment
# 3	120+ days delinquent
# 4	90 days delinquent
# 5	60 days delinquent
# 6	30 days delinquent
# 7	unknown delinquency
# 8	current and never delinquent
# 9	all other

#%% Data exploration

#Exclude records with missing values
df = df[df['ExternalRiskEstimate']!=-9] 

# Data summary 
## Descriptive stats for numerical variables
df_summary = df[['Class'] + features].describe()
## Descriptive stats for categorical variables
df_summary2 = df.groupby(var_category)[var_category].count()
## Group mean values
df_summary3 = df.groupby('Class').mean()
df_summary4 = df.groupby('ExternalRiskEstimate').mean()
## Correlations for numerical variables
df_corr = df[['Class'] + features].corr() #Point Biserial Correlation btw Class and feature variables
## Top correlated feature variables
df_corr2 = pd.DataFrame({'corr': df_corr['Class'], 'abs_corr': abs(df_corr['Class'])})
df_corr2.sort_values(by = 'abs_corr', ascending=False, inplace=True)
features_sub = list(df_corr2.iloc[1:16].index)

# Top features for considerations (correlations in parenthesis):
# ExternalRiskEstimate	-0.46
# NetFractionRevolvingBurden	0.33
# PercentTradesWBalance	0.28
# PercentTradesNeverDelq	-0.26
# AverageMInFile	-0.25
# MaxDelq2PublicRecLast12M	-0.24
# MaxDelqEver	-0.22
# MSinceOldestTradeOpen	-0.20
# NumTrades60Ever2DerogPubRec	0.17
# NumInqLast6M	0.14
# NumSatisfactoryTrades	-0.14
# PercentInstallTrades	0.14
# NumTrades90Ever2DerogPubRec	0.14
# NumInqLast6Mexcl7days	0.14
# MSinceMostRecentInqexcl7days	-0.12


# Correlation matrix for select variables
df_corr3 = df[features_sub].corr() 

### Output to excel
with pd.ExcelWriter('output_data_summary.xlsx') as writer:  
    df_summary.to_excel(writer, sheet_name='Summary_Numeric')
    df_summary2.to_excel(writer, sheet_name='Summary_Category')
    df_summary4.to_excel(writer, sheet_name='Summary_Group_Mean')
    df_corr.to_excel(writer, sheet_name='Corr1')
    df_corr2.to_excel(writer, sheet_name='Corr2')
    df_corr3.to_excel(writer, sheet_name='Corr3')

## Other correlation methods
# from scipy import stats
# stats.f_oneway(df.loc[df[target_variable]==target_names[0], features[0]],
#                df.loc[df[target_variable]==target_names[1], features[0]]) #ANOVA when Class has two or more values
# stats.pearsonr(df[features[0]], df[features[1]]) #Pearson’s Correlation
# stats. spearmanr(df[features[0]], df[features[1]]) #Spearman’s Correlation


#%% Data visualization

df2 = df[[target_variable] + features_sub].copy() #Data subset for plotting

# Box plot
sns.boxplot(x=features_sub[0], y=target_variable, data=df)
## Single boxplot with multiple subplots for each variable
n_rows = 5
n_cols = 3
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 20))
for i in list(range(0, n_rows)):
    for j in list(range(0, n_cols)):
        k = (i*n_cols)+j
        if k<len(features_sub):
            sns.boxplot(x=features_sub[k], y=target_variable, data=df, ax=axes[i, j])


#Structure multi-plots based on groups
col1 = "MaxDelq2PublicRecLast12M"
var_x = "ExternalRiskEstimate"
var_y = "PercentTradesNeverDelq"

g = sns.FacetGrid(df, col=col1, col_wrap=3, height=3, hue=target_variable)
g.map(plt.scatter, var_x, var_y,  alpha=.7)
g.add_legend();


#Pairwise plot
vars1 = ['ExternalRiskEstimate', 'NetFractionRevolvingBurden', 'AverageMInFile']

g = sns.PairGrid(df[vars1+[target_variable]], hue=target_variable)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend();


#%% User-defined functions

# Print summary for out-of-sample model perforamce
def model_summary(y_test, dtc_predict, dtc_cv_score):
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

# Search for best depth parameter for decision tree model
def tree_depth_search (X_train, X_test, y_train, y_test, criterion, depth_range):
    dt2 = pd.DataFrame()
    for i in depth_range:
        
        clf = tree.DecisionTreeClassifier(criterion=criterion, max_depth=i, random_state=66)
        clf = clf.fit(X_train,y_train)
        out_sample_pred = clf.predict(X_test)
        in_sample_pred = clf.predict(X_train)
        in_acc = sk.metrics.accuracy_score(y_train, in_sample_pred)
        out_acc = sk.metrics.accuracy_score(y_test, out_sample_pred)
        in_conf = sk.metrics.confusion_matrix(y_train, in_sample_pred)
        out_conf = sk.metrics.confusion_matrix(y_test, out_sample_pred)
        tpr_in = in_conf[1][1]/(in_conf[1][0]+in_conf[1][1])
        tpr_out = out_conf[1][1]/(out_conf[1][0]+out_conf[1][1])
        fpr_in = in_conf[0][1]/(in_conf[0][0]+in_conf[0][1])
        fpr_out = out_conf[0][1]/(out_conf[0][0]+out_conf[0][1])
        
        dt1 = pd.DataFrame({'Depth': [i, i], 'Sample': ['In-Sample', 'Out-Sample'], 'Accuracy': [in_acc, out_acc],
                            "TPR": [tpr_in,tpr_out], "FPR": [fpr_in, fpr_out]})
        dt2 = pd.concat([dt2, dt1])
   
    return dt2


#%% Logisitic Regressions

#Define predcitors and target variable
X = df[features_sub].copy() 
y = df['Class'].copy()

# Implement train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)


from sklearn.feature_selection import RFE

lge = LogisticRegression(solver='liblinear', random_state=66)
lge = RFE(lge, 20)
lge = lge.fit(X_train, y_train)

print(lge.support_)
print(lge.ranking_)


lge_predict = lge.predict(X_test) # predictions
lge_cv_score = cross_val_score(lge, X, y, cv=10, scoring='roc_auc') #scores
model_summary(y_test, lge_predict, lge_cv_score) # Model summary


import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train)
logit_model_result=logit_model.fit()
print(logit_model_result.summary2())

# ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, lge.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, lge.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#%% Decision tree model

# Preliminary model fit
dtc = tree.DecisionTreeClassifier(max_depth=5)
dtc = dtc.fit(X_train, y_train)
dtc_predict = dtc.predict(X_test) # predictions
dtc_cv_score = cross_val_score(dtc, X, y, cv=10, scoring='roc_auc') #scores
model_summary(y_test, dtc_predict, dtc_cv_score) # Model summary

## Plot the tree
dot_data = tree.export_graphviz(dtc, feature_names=features, class_names=target_names, out_file=None, filled = True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
# graph.write_png("tree_model1.png")


# Model optimization
dtc_depth = tree_depth_search(X_train, X_test, y_train, y_test, criterion='entropy', depth_range=range(1, 12))

## Plot the model measures
dt1 = pd.melt(dtc_depth, id_vars=['Depth', 'Sample'], var_name='Measure') #Reshape data
### Method 1
g = sns.FacetGrid(dt1, col='Measure', hue='Sample')
g.map(plt.scatter, "Depth", "value",  alpha=.7)
g.add_legend(title=None);
### Method 2
sns.relplot(x="Depth", y="value", hue="Sample", col="Measure", aspect=.75, linewidth=2.5, kind="line", data=dt1)


# Refit the model using tuned parameters
dtc = tree.DecisionTreeClassifier(max_depth=4)
dtc = dtc.fit(X_train, y_train)
dtc_predict = dtc.predict(X_test) # predictions
dtc_cv_score = cross_val_score(dtc, X, y, cv=10, scoring='roc_auc') #scores
model_summary(y_test, dtc_predict, dtc_cv_score) # Model summary

## Plot the tree
dot_data = tree.export_graphviz(dtc, feature_names=features, class_names=target_names, out_file=None, filled = True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
# graph.write_png("tree_model2.png")


#%% Create a Random Forest Model

# Preliminary model fit
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train) #fit
rfc_predict = rfc.predict(X_test) # predictions
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc') #scores
model_summary(y_test, rfc_predict, rfc_cv_score) # Model summary

# Model optimization
## Optimize select hyperparamaters
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)] # number of trees in random forest
max_features = ['auto', 'sqrt'] # number of features at every split
max_depth = [int(x) for x in np.linspace(100, 500, num = 11)] # max depth
max_depth.append(None)

## Create random grid
random_grid = {
 'n_estimators': n_estimators,
 'max_features': max_features,
 'max_depth': max_depth
 }

## Random search of parameters
rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, 
                                cv = 3, verbose=2, random_state=66, n_jobs = -1)
rfc_random.fit(X_train, y_train) # Fit the model
print(rfc_random.best_params_) # Print results


# Refit the model using tuned parameters
rfc = RandomForestClassifier(n_estimators=944, max_depth=100, max_features='auto')
rfc.fit(X_train,y_train) #fit
rfc_predict = rfc.predict(X_test) # predictions
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc') #scores
model_summary(y_test, rfc_predict, rfc_cv_score) # Model summary


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

