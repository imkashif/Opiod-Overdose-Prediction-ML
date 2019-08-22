import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score

ps = pd.read_csv("Dataset/prescriber.csv")

############################################################################################
ps=pd.get_dummies(ps, prefix=['D'], columns=['State'])
ps=pd.get_dummies(ps, prefix=['D'], columns=['Specialty'])
ps=pd.get_dummies(ps, prefix=['D'], columns=['Gender'])
y=ps['Opioid.Prescriber'].values

my_cols=set(ps.columns)
my_cols.remove('Credentials')
ps=ps[my_cols]

X=ps.loc[:,].values
############################################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#X_test_df=pd.DataFrame(data=X_test[1:,1:],index=X_test[1:,0],columns=X_test[0,1:])
#X_test_df.to_csv('Dataset/pres_test1.csv')

learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
    gb.fit(X_train, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_test, y_test)))

    
gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.5, max_features=2, max_depth = 2, random_state = 0)
model=gb.fit(X_train, y_train)

y_pred=model.predict(X_test)
score = accuracy_score(y_test, y_pred)

pred = model.predict_proba(X_test)
print(pred[:,1])

#import pickle
#creating and training a model
#serializing our model to a file called model.pkl
#pickle.dump(model, open("model2.pkl","wb"))
