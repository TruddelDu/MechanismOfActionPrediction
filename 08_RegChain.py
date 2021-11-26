import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import RegressorChain
import xgboost as xgb
from sklearn.metrics import log_loss
import logging
import time

## PREPROCESSING
logging.basicConfig(level=logging.INFO)
start = time.perf_counter()

X_data = pd.read_csv('MechanismOfAction/train_features.csv')
y_data = pd.read_csv('MechanismOfAction/train_targets_scored.csv')

# delete all unneeded columns
X_data.drop(columns=['sig_id', 'cp_type', 'c-42', 'c-4', 'c-13', 'c-94', 'c-2', 'c-31'], inplace=True) #, 'c-42', 'c-4', 'c-13', 'c-94', 'c-2', 'c-31'
y_data.drop(columns=['sig_id'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.2, 
                                                    random_state=174)
logging.info(f'Length of X_train: {len(X_train)}, {type(X_train)}')

# no missing values present in data

# Labelencoding cp_dose
label_encoder = LabelEncoder()
for col in ['cp_dose']:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])



## XGBoost MODEL
logging.info('Training model')
regressor_chain = RegressorChain(xgb.XGBRegressor(tree_method='gpu_hist', objective='reg:logistic', eval_metric='logloss'), random_state=174).fit(X_train, y_train)


logging.info('Training model finished')


y_train_pred = regressor_chain.predict(X_train)
y_test_pred = regressor_chain.predict(X_test)

logging.info(f'training log loss: {log_loss(y_train, y_train_pred)}')
logging.info(f'test log loss : {log_loss(y_test, y_test_pred)}')
end = time.perf_counter()
logging.info(f'runtime: {round((end-start)/60,1)} m')
# INFO:root:training log loss: 0.31634226851778946
# INFO:root:test log loss : 2.7886626381803423