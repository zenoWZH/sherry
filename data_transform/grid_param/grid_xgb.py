import numpy as np
import pandas as pd
import warnings
import json

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score, ShuffleSplit

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from xgboost import plot_importance

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

param_dist = {
        'n_estimators':range(20,100,5),
        'max_depth':range(2,10,1),
        'learning_rate':[0.01,0.1],
        #'subsample':[1],
        #'colsample_bytree':[1],
        'min_child_weight':range(1,9,1),
        #'gpu_id':[0],
        #'tree_method':['gpu_hist']
        }

clf = GridSearchCV(XGBClassifier(), param_dist, cv=ShuffleSplit(5, test_size = .2, train_size = .8), scoring='%s_macro' % score, n_jobs= 8)#, verbose=10)

clf.fit(X, Y)

print(clf.best_params_)
#with open("bestparams_xgb.json", "wb") as xgbjs:
#       json.dump(clf.best_params_, xgbjs)
