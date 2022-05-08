import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score, ShuffleSplit

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from xgboost import plot_importance

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

df = pd.read_csv("NewData.csv")
data =df.values

#scaler = MinMaxScaler()
scaler = StandardScaler()
result_feature = scaler.fit_transform(data[:,:19])
result_label = data[:,19]
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
        'n_estimators':range(10,100,5),
        'max_depth':range(2,15,1),
        'learning_rate':np.linspace(0.2,1,5),
        'subsample':np.linspace(0.6,0.9,10),
        'colsample_bytree':np.linspace(0.5,0.98,10),
        'min_child_weight':range(1,9,1),
        #'gpu_id':[0],
        #'tree_method':['gpu_hist'],
        #"predictor":["gpu_predictor"]
        }

clf = GridSearchCV(XGBClassifier(), param_dist, cv=ShuffleSplit(10, test_size = .1, train_size = .9), scoring='%s_macro' % score, n_jobs= 5)

clf.fit(X, Y)

print(clf.best_params_)