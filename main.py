from tkinter import *
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
import joblib

loaded_model = xgb.Booster()
loaded_model.load_model('model.txt')

hosp = pd.read_csv('C:/Users/Cordell Brettle/Documents/CS Lab/Paper 3/2016/coreElective2016_Half_1.csv', low_memory=False)

hosp.fillna(0)

hosp = hosp[~hosp.isin([np.nan, np.inf, -np.inf]).any(1)]

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

def testPatient():
    #hosp.at[hosp.iloc[(hosp['fName'] == nameEntryF.get()) & (hosp['lName'] == nameEntryL.get()) & (hosp['Birthdate'] == bDayEntry.get())], 'I10_PR1']

    #Xnew = hosp.iloc[(hosp['fName'] == nameEntryF.get()) & (hosp['lName'] == nameEntryL.get()) & (hosp['Birthdate'] == bDayEntry.get()), :-1].values
    Xnew = hosp.iloc[0:1, :-1]
    print(Xnew)
    dnew = xgb.DMatrix(Xnew, enable_categorical=True)

    print(dnew.get_label())

    predsNew = loaded_model.predict(dnew)
    print(predsNew)
    resultLabel.config(text = '0')


root = Tk()

nameLabelF = Label(root, text="First Name: ")
nameLabelL = Label(root, text="Last Name: ")
bDayLabel = Label(root, text="Birthdate")
procedureLabel = Label(root, text="Procedure Code: ")
resultLabel = Label(root, text="")
nameEntryF = Entry()
nameEntryL = Entry()
bDayEntry = Entry()
procedureEntry = Entry()
enter = Button(text="Enter", command = testPatient)

nameLabelF.grid(row=0, column=0)
nameLabelL.grid(row=1, column=0)
nameEntryF.grid(row=0, column=1)
nameEntryL.grid(row=1, column=1)
bDayLabel.grid(row=2, column=0)
bDayEntry.grid(row=2, column=1)
procedureLabel.grid(row=3, column=0)
procedureEntry.grid(row=3,column=1)
enter.grid(row=4,column=0)
resultLabel.grid(row=4,column=1)

root.mainloop()
