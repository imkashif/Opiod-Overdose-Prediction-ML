# Load libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib as mplot
import IPython
from IPython.core.display import HTML
from IPython.core.debugger import set_trace
from distutils.version import StrictVersion
import xlrd
print("numpy version:  %s" % np.__version__)
print("pandas version:  %s" % pd.__version__)
print("matplotlib version:  %s" % mplot.__version__)
print("IPython version:  %s" % IPython.__version__)
print("seaborn version:  %s" % sns.__version__)

if StrictVersion(np.__version__) >= StrictVersion('1.13.0') and \
   StrictVersion(pd.__version__) >= StrictVersion('0.20.0') and \
   StrictVersion(mplot.__version__) >= StrictVersion('2.0.0') and \
   StrictVersion(IPython.__version__) >= StrictVersion('5.5.0') and \
   StrictVersion(sns.__version__) >= StrictVersion('0.7.0'):
    print('\nCongratulations, your environment is setup correctly!')
else:
    print('\nEnvironment is NOT setup correctly!')

# load scikit-learn modules
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split


df = pd.read_pickle('Dataset/data_cleaned_oh.pkl')

#df.to_csv('Dataset/output.csv')

df.head()

# Let's use the following variables as our initial set of predictors
cat_cols = ['gender', 'marital', 'race', 'ethnicity']
cat_cols_encoded = [c + '_encoded' for c in cat_cols]
numeric_cols = ['prior_opioid_abuse_diag', 'age', 'opioid_discharge_days_supply']
pred_cols = numeric_cols + cat_cols_encoded
target_col = 'overdose'
all_cols = cat_cols+numeric_cols+[target_col]

df_opioids = df[df['prescribed_opioids'] == 1]

# Encode the categorical variables
dfe = df_opioids[cat_cols]

# Replace missing data with an 'Unknown' category
# so the missing data will also be encoded
dfe = dfe.replace(np.NaN,'Unknown')

# Encode the categorical variables
encoded = dfe.apply(preprocessing.LabelEncoder().fit_transform)

# Append the non-categorical variables and the encoded variables
# into a single Dataframe
# Name the new variables as <name>_encoded
dfe = pd.concat([df_opioids[all_cols], encoded.add_suffix('_encoded')],axis=1)
dfe.head(2)

# let's build a model using this set of variables...
pred_cols = ['age', 'opioid_discharge_days_supply',
             'prior_opioid_abuse_diag', \
             'gender_encoded', 'marital_encoded', 'race_encoded', \
             'ethnicity_encoded']
#pred_cols = ['prior_abuse_diag', 'adult', 'age_at_visit', \
#             'opioid_discharge_days_supply', 'gender_encoded', \
#             'marital_encoded', 'race_encoded', 'ethnicity_encoded']

LR_pred_cols = pred_cols
X = dfe[pred_cols].to_numpy()
y = dfe['overdose'].to_numpy()
print('Using predictor variables of:',pred_cols)

# models we'll consider
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Let's try a simple logistic regression model to see how predictive our data is

# perform a model fit on the training set
LR = LogisticRegression()
result = LR.fit(X, y)

# calculate predicted values from the model to compare with actual outcomes
expected = y
predicted = LR.predict(X)

print('\nClassification Report\n',metrics.classification_report(expected, predicted))
print('\nConfusion Matrix\n',metrics.confusion_matrix(expected, predicted))
print('\nAccuracy score =',metrics.accuracy_score(expected, predicted))
print('\nAUC score =',metrics.roc_auc_score(expected, predicted))
print('\nf1 score =',metrics.f1_score(expected, predicted))

# Lets graph an ROC curve for the model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# we'll use the test set (rather than training) for this evaluation
auc = roc_auc_score(y, LR.predict(X))
probs = LR.predict_proba(X)[:,1]
fpr, tpr, thresholds = roc_curve(y, probs)
tpr[1] = tpr[0]

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Model')
plt.legend(loc="lower right")
plt.show()


# Create the training and test datasets
# Partition 30% of the data to the Test set
train, test = train_test_split(dfe, test_size=0.3, random_state=987)

X_train = train[pred_cols].to_numpy()
y_train = train['overdose'].to_numpy()
X_test = test[pred_cols].to_numpy()
y_test = test['overdose'].to_numpy()

print('X_train shape = ',X_train.shape)
print('y_train shape = ',y_train.shape)
print('X_test shape = ',X_test.shape)
print('y_test shape = ',y_test.shape)


# Create Decision Tree models with tree depths of 1 to 25
from sklearn.metrics import roc_curve, auc
tree_depths = np.linspace(1, 25, 25, endpoint=True)
train_auc = []
test_auc = []
for d in tree_depths:
    dt = DecisionTreeClassifier(max_depth=d)
    dt.fit(X_train, y_train)  # Fit a model to a tree at the current tree depth
    # Calc the false positive rate and the true positive rates by comparing the training answers to the model
    pred = dt.predict(X_train)
    # Compute the AUC from the rates and append it to the training auc data
    fpr, tpr, thresholds = roc_curve(y_train, pred)
    train_auc.append(auc(fpr, tpr))
    # Calc the false positive rate and the true positive rates by comparing the test answers to the model
    pred = dt.predict(X_test)
    # Compute the AUC from the rates and append it to the test auc data
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    test_auc.append(auc(fpr, tpr))

# Graph the AUCs of the training and test models at various tree depths
# Notice that the training scores continue to improve until they hit 1.0 (Perfect model, complete memorization)
# But the test scores get worse as the training model improves
from matplotlib import pyplot as plt

plt.plot(tree_depths, train_auc, 'g', label='Train AUC')
plt.plot(tree_depths, test_auc, 'r', label='Test AUC')
plt.legend(loc=3)
plt.ylim((0.5, 1.01))
plt.ylabel('AUC')
plt.xlabel('Tree Depth')
plt.show()

# Split the data using KFold cross validation
kfold = model_selection.KFold(n_splits=3, random_state=123)
data = list(range(0, 9))
print("Data:", data)
for train, test in kfold.split(data):
    print("Train: ", train, 'Test:', test)

# Perform k-fold cross validation
LR = LogisticRegression(solver='liblinear')

result = LR.fit(X, y)
print(LR)

kfold = model_selection.KFold(n_splits=10)
results = model_selection.cross_val_score(LR, X, y, cv=kfold, scoring='roc_auc')

print('Model score = %.4f (%.4f)' % (results.mean(), results.std()))


# Bootstrap with replacement

data = np.array(range(0,9))
print("Data:",data)

bootstrap = []
for i in [0,3]:  # 3 splits, 70% sample
    indexes = np.random.choice(len(data),int(0.7*len(data)),replace=True)
    bootstrap.append(list(data[indexes]))

for train in bootstrap:
    print("Train: ", train)



# Perform bootstrap (with replacement)
from sklearn.ensemble import BaggingClassifier

bootstrap = BaggingClassifier(LogisticRegression(solver='liblinear'), n_estimators=3, max_samples=0.7,
                              bootstrap=True, random_state=123)
fit = bootstrap.fit(X,y)
print(bootstrap)

score = bootstrap.score(X,y)
print("Model score = ",score)


# Perform bootstrap (without replacement)
LR = LogisticRegression(solver='liblinear')

result = LR.fit(X,y)
print(LR)

bootstrap = model_selection.ShuffleSplit(10, test_size=0.3)
results = model_selection.cross_val_score(LR, X, y, cv=bootstrap, scoring='roc_auc')

print('Model score = %.4f (%.4f)' %(results.mean(), results.std()))


# prepare configuration for cross validation test harness
# (from https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/)
seed = 123
# prepare models
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier(n_estimators=10)))
models.append(('NB', GaussianNB()))

# evaluate each model in turn
results = []
names = []
scoring = 'roc_auc'
# others include: 'accuracy', 'f1', 'roc_auc',
# or found here: http://scikit-learn.org/stable/modules/model_evaluation.html

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train,
                                                 cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
plt.ylabel(scoring)
ax.set_xticklabels(names)
plt.show()

# evaluate each model in turn
results = []
names = []

####### EDIT HERE #######
# scoring = '???'
#########################

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train,
                                                 cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
plt.ylabel(scoring)
ax.set_xticklabels(names)
plt.show()

# use Logistic Regression & Random Forests
models = []

####### EDIT HERE #######
#models.append(('LR', LogisticRegression(solver='liblinear')))
#models.append(('RF', RandomForestClassifier(n_estimators=10)))
#########################

# evaluate each model in turn
results = []
names = []

####### EDIT HERE #######
#scoring = '?????'
#########################

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train,
                                                 cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


### automated grid search
from sklearn.model_selection import GridSearchCV

param_grid = [
        {'n_estimators': [50, 100, 250],
         'class_weight': [None, 'balanced'],
         'max_features': [2, 'sqrt', None]}
]

rfc = RandomForestClassifier()
grid_search = GridSearchCV(rfc, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)
cvres = grid_search.cv_results_


# print results
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(round(np.sqrt(mean_score), 3), params)


# assign parameters from best fit
final_fit = RandomForestClassifier(class_weight=None,
                                   max_features=2,
                                   n_estimators=100)

final_fit.fit(X_train, y_train)

# store predicted values using the final model
pred_train = final_fit.predict(X_train)
pred_test = final_fit.predict(X_test)

# explore performance on training data
pd.crosstab(y_train, pred_train,
            rownames=["Actual"], colnames=["Predicted"])

# explore performance on testing data
pd.crosstab(y_test, pred_test,
            rownames=["Actual"], colnames=["Predicted"])

# Show how to use the resulting model to predict opioid overdose
# age, opioid_discharge_days_supply, prior_opioid_abuse_diag,
# gender (F), marital (M), race (white), ethnicity (english)
#new_patient = [45,10,1,0,1,4,7]

#pred = final_fit.predict(np.asmatrix(new_patient))
#if pred[0] == 0:
#    print('Patient has no overdose risk.')
#elif pred[0] == 1:
#    print('Patient has overdose risk.')

prob=final_fit.predict_proba(X_test)[:,1]
print("precentage",prob)



