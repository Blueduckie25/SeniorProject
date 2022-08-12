import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report

hosp = pd.read_csv('C:/Users/Cordell Brettle/Downloads/archive (5)/data01.csv')

hosp = hosp.dropna()


X = hosp.iloc[:, :-1]
Y = hosp.iloc[:, -1]
#labelencoder_X = LabelEncoder()

#X[:, 0] = labelencoder_X.fit_transform(X[:,0])
#onehotencoder = OneHotEncoder(categories="auto")
#X = onehotencoder.fit_transform(X).toarray()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(x_test, label=y_test, enable_categorical=True)


param = {
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'learning_rate': 0.5,
    'objective': 'binary:logistic',
    'max_depth': None,
    'tree_method' : "gpu_hist",
    'n_jobs': 1
}

steps = 200

xg_reg = xgb.train(param, dtrain, steps, evals=[(dtrain, 'train')])

xg_reg.save_model('model.txt')

preds = xg_reg.predict(dtest)
best_preds = np.asarray([np.argmax(line) for line in preds])

matrix = confusion_matrix(y_test, best_preds)

print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
print("AUC = {}".format(roc_auc_score(y_test, preds)))



