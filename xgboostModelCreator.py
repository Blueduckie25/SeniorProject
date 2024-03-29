import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import sklearn.preprocessing as preprocessing
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

hosp = pd.read_csv('C:/Users/Cordell Brettle/Documents/CS Lab/Paper 3/2016/coreElective2016_Half_1.csv', low_memory=False)

hosp.fillna(0)

hosp = hosp[~hosp.isin([np.nan, np.inf, -np.inf]).any(1)]

#hosp.DIED = hosp.DIED.astype('bool')

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


X = hosp.iloc[:, :-1]
Y = hosp.iloc[:, -1]

#oversample = SMOTE()
#X, Y = oversample.fit_resample(X, Y)

#labelencoder_X = LabelEncoder()

#X[:, 0] = labelencoder_X.fit_transform(X[:,0])
#onehotencoder = OneHotEncoder(categories="auto")
#X = onehotencoder.fit_transform(X).toarray()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)

print(len(y_train[y_train==1]))

dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(x_test, label=y_test, enable_categorical=True)


param = {
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'learning_rate': 0.5,
    'objective': 'binary:logistic',
    'max_depth': None,
    'n_estimators': 500,
    'tree_method' : "gpu_hist",
    'n_jobs': 1
}

steps = 200

xg_reg = xgb.train(param, dtrain, steps, evals=[(dtrain, 'train')])

xg_reg.save_model('model.txt')

preds = xg_reg.predict(dtest)
best_preds = np.asarray([np.argmax(line) for line in preds])

matrix = confusion_matrix(y_test, best_preds)

print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
print("AUC = {}".format(roc_auc_score(y_test, preds)))

print(matrix)

