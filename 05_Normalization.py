import pandas as pd 
from sklearn.model_selection import train_test_split
import time
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from sklearn.metrics import log_loss

start = time.perf_counter()
logging.basicConfig(level=logging.DEBUG)

X_data = pd.read_csv('MechanismOfAction/train_features.csv')
y_data = pd.read_csv('MechanismOfAction/train_targets_scored.csv')

y_data.drop(columns=['sig_id'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.2, 
                                                    random_state=174)


controls = X_train[X_train['cp_type'] == 'ctl_vehicle']
normalized_X_train = list()
normalized_X_test = list()
for dose in controls.cp_dose.unique():
    for duration in controls.cp_time.unique():
        specific_controls = controls[(controls.cp_dose==dose) & (controls.cp_time==duration)]
        train_tmp = X_train[(X_train.cp_dose==dose) & (X_train.cp_time==duration)].copy()
        normalized_X_train.append(train_tmp - specific_controls.mean())

        test_tmp = X_test[(X_test.cp_dose==dose) & (X_test.cp_time==duration)].copy()
        normalized_X_test.append(test_tmp - specific_controls.mean())

normalized_X_train = pd.concat(normalized_X_train)
normalized_X_test = pd.concat(normalized_X_test)


# Labelencoding cp_dose
label_encoder = LabelEncoder()
for col in ['cp_dose']:
    normalized_X_train[col] = label_encoder.fit_transform(normalized_X_train[col])
    normalized_X_test[col] = label_encoder.transform(normalized_X_test[col])


normalized_X_train.drop(columns=['sig_id', 'cp_type'], inplace=True) #, 'c-42', 'c-4', 'c-13', 'c-94', 'c-2', 'c-31'
normalized_X_test.drop(columns=['sig_id', 'cp_type'], inplace=True) #, 'c-42', 'c-4', 'c-13', 'c-94', 'c-2', 'c-31'

## XGBoost MODEL
logging.info('Training model')
multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(tree_method='gpu_hist', objective='reg:logistic', eval_metric='logloss')).fit(normalized_X_train, y_train)


logging.info('Training model finished')


y_train_pred = multioutputregressor.predict(normalized_X_train)
y_test_pred = multioutputregressor.predict(normalized_X_test)

logging.debug(f'training log loss: {log_loss(y_train, y_train_pred)}')
logging.debug(f'test log loss : {log_loss(y_test, y_test_pred)}')

end = time.perf_counter()
logging.debug(f'runtime: {round((end-start)/60,1)} m')

print('DEBUG:root:training log loss: 0.1683120116454378 \
        DEBUG:root:test log loss : 3.9309048451250175   \
        DEBUG:root:runtime: 17.7 m')

#Normalization with controls reduces accuracy

