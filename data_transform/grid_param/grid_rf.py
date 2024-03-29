import numpy as np
import pandas as pd
import warnings
import json

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score, ShuffleSplit

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

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

score = 'f1'

param_dist = {'bootstrap': [True],
              'max_depth': range(5,25,2),
              'max_features': range(3,16),
              'min_samples_leaf': [1, 2, 4, 6, 8],
              'min_samples_split': [2, 3, 4, 5, 6, 7, 8],
              'n_estimators': range(20,80,5)}

with open("./bestparams_rf.json", "w") as xgbjs:
       for i in range(10):
              clf = GridSearchCV(RandomForestClassifier(), param_dist, cv=ShuffleSplit(5, test_size = .2, train_size = .8), scoring='%s_macro' % score, n_jobs= 16)#, verbose=10)

              clf.fit(X, Y)

              print(clf.best_params_)
              
              json.dump(clf.best_params_, xgbjs)
              xgbjs.write("\n")

xgbjs.close()
