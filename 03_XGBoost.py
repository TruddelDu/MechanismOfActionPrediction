import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from sklearn.metrics import log_loss
import logging

## PREPROCESSING
logging.basicConfig(level=logging.DEBUG)

X_data = pd.read_csv('MechanismOfAction/train_features.csv')
y_data = pd.read_csv('MechanismOfAction/train_targets_scored.csv')

# delete all unneeded columns
X_data.drop(columns=['sig_id', 'cp_type', 'c-42', 'c-4', 'c-13', 'c-94', 'c-2', 'c-31'], inplace=True) #, 'c-42', 'c-4', 'c-13', 'c-94', 'c-2', 'c-31'
y_data.drop(columns=['sig_id'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.2, 
                                                    random_state=174)
logging.debug(f'Length of X_train: {len(X_train)}, {type(X_train)}')

# no missing values present in data

# Labelencoding cp_dose
label_encoder = LabelEncoder()
for col in ['cp_dose']:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])



## XGBoost MODEL
logging.info('Training model')
multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(tree_method='gpu_hist', objective='reg:logistic', eval_metric='logloss')).fit(X_train, y_train)


logging.info('Training model finished')


y_train_pred = multioutputregressor.predict(X_train)
y_test_pred = multioutputregressor.predict(X_test)

logging.debug(f'training log loss: {log_loss(y_train, y_train_pred)}')
logging.debug(f'test log loss : {log_loss(y_test, y_test_pred)}')


print('DEBUG:root:training log loss: 0.16495482186892862    \
    DEBUG:root:test log loss : 2.727398407356135           \
    20 min runtime') # deleting cp_type has little to no impact on the results and neither do the highly correlated features