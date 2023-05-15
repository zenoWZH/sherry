import numpy as np
import pandas as pd
import warnings
import json

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score, ShuffleSplit

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

df = pd.read_csv("../../Boxcox_data.csv")
data =df.values

Y = df["Outcome"].values
X = df[['Gender', 'Age', 'Height', 'Weight', 'BMI', 'Hypertension',
       'SBP', 'DBP', 'PR', 'Drink', 'Smoke', 'FPG', 'AST', 'ALT', 'BUN', 'Scr',
       'TG', 'TC']].values
names = ['Gender', 'Age', 'Height', 'Weight', 'BMI', 'Hypertension',
       'SBP', 'DBP', 'PR', 'Drink', 'Smoke', 'FPG', 'AST', 'ALT', 'BUN', 'Scr',
       'TG', 'TC']



score = 'f1_macro'

param_dist = {'C': [0.1, 1, 5, 10, 20, 50, 80, 100, 200, 500, 1000], 
              'gamma': ['scale', 'auto'],
              #'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
              'kernel': ['linear', 'poly']}

with open("./bestparams_svm.json", "a") as xgbjs:
       for i in range(10):
              clf = GridSearchCV(SVC(), param_dist, cv=ShuffleSplit(5, test_size = .2, train_size = .8), scoring='%s' % score, n_jobs= 16)#, verbose=10)

              clf.fit(X, Y)

              print(clf.best_params_)
              
              json.dump(clf.best_params_, xgbjs)
              xgbjs.write("\n")

xgbjs.close()
