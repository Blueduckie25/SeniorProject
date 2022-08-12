import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import pickle

hosp = pd.read_csv('C:/Users/Cordell Brettle/Documents/CS Lab/Paper 3/2016/coreElective2016_Half_1.csv', low_memory=False)

hosp = hosp.drop(columns=['fName', 'lName', 'Birthdate'])

hosp.fillna(0)

hosp = hosp[~hosp.isin([np.nan, np.inf, -np.inf]).any(1)]

hosp.DIED = hosp.DIED.astype('bool')

hosp.AGE = hosp.AGE.astype('category')
hosp.FEMALE = hosp.FEMALE.astype('category')
hosp.HOSPST = hosp.HOSPST.astype('category')
hosp.I10_DX1 = hosp.I10_DX1.astype('category')
hosp.I10_PR1 = hosp.I10_PR1.astype('category')
hosp.PSTATE = hosp.PSTATE.astype('category')

hosp.AGE = hosp.AGE.cat.codes
hosp.FEMALE = hosp.FEMALE.cat.codes
hosp.HOSPST = hosp.HOSPST.cat.codes
hosp.I10_DX1 = hosp.I10_DX1.cat.codes
hosp.I10_PR1 = hosp.I10_PR1.cat.codes
hosp.PSTATE = hosp.PSTATE.cat.codes

print(hosp)

X = hosp.iloc[:, :-1]
Y = hosp.iloc[:, -1]

oversample = SMOTE()
X, Y = oversample.fit_resample(X, Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=1, stratify=Y)

model = RandomForestClassifier()

model.fit(x_train, y_train)

preds = model.predict(x_test)

print("Precision = {}".format(precision_score(y_test, preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, preds)))
print("AUC = {}".format(roc_auc_score(y_test, preds)))

matrix = confusion_matrix(y_test, preds)
print(matrix)

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

