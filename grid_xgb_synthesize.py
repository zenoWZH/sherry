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

df = pd.read_csv("./data_synthesize/out/correlated_attribute_mode/sythetic_data.csv")
data =df.values

#scaler = MinMaxScaler()
scaler = StandardScaler()
result_feature = scaler.fit_transform(data[:,:18])
result_label = data[:,18]
result = np.append(result_feature, result_label.reshape(len(result_label),1), axis = 1)
df_newdata = pd.DataFrame(result, columns= df.columns)


Y = df["Outcome"].values
X = df[['Gender', 'Age', 'Height', 'Weight', 'BMI', 'Hypertension',
       'SBP', 'DBP', 'PR', 'Drink', 'Smoke', 'FPG', 'AST', 'ALT', 'BUN', 'Scr',
       'TG', 'TC']].values
names = ['Gender', 'Age', 'Height', 'Weight', 'BMI', 'Hypertension',
       'SBP', 'DBP', 'PR', 'Drink', 'Smoke', 'FPG', 'AST', 'ALT', 'BUN', 'Scr',
       'TG', 'TC']


score = 'f1'

param_dist = {
        'n_estimators':range(20,80,5),
        'max_depth':range(2,20,1),
        'learning_rate':[0.01,0.1],
        'subsample':[1],
        #'colsample_bytree':[1],
        'min_child_weight':range(1,9,1),
        #'gpu_id':[0],
        #'tree_method':['gpu_hist']
        }

clf = GridSearchCV(XGBClassifier(), param_dist, cv=ShuffleSplit(5, test_size = .2, train_size = .8), scoring='%s_macro' % score, n_jobs= 6)#, verbose=10)

clf.fit(X, Y)

print(clf.best_params_)
#with open("bestparams_xgb.json", "wb") as xgbjs:
#       json.dump(clf.best_params_, xgbjs)
