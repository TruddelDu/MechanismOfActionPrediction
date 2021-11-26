import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from dask.distributed import Client
# import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from sklearn.metrics import log_loss
import logging
import time


logging.basicConfig(level=logging.DEBUG)
# client = Client(process=False)      # create local cluster

logging.info(f'start @ {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
## PREPROCESSING
start = time.perf_counter()

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
multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(tree_method='gpu_hist', eval_metric='logloss'))

params = {
    # 'estimator__eta' : [0.2, 0.3, 0.5], #learning_rate
    'estimator__objective' : ['reg:sqaurederror', 'reg:logistic'],
    'estimator__max_depth' : [10, 15],
#    'estimator__scale_pos_weight' : [1, 0.5, 2],
    # 'estimator__max_delta_step' : [1]
}

# logging.info(sorted(multioutputregressor.get_params().keys()))



logging.info('Training model')
search = GridSearchCV(multioutputregressor, params, scoring='neg_log_loss' ,verbose=2, cv=5, n_jobs=-2)
# with joblib.parallel_backend('dask'):
search.fit(X_train, y_train)



logging.info('Training model finished')


y_train_pred = search.predict(X_train)
y_test_pred = search.predict(X_test)

logging.info(f'training log loss: {log_loss(y_train, y_train_pred)}')
logging.info(f'test log loss : {log_loss(y_test, y_test_pred)}')
logging.info(search.best_params_)
end = time.perf_counter()
logging.info(f'runtime: {round((end-start)/60,1)} m')

print('INFO:root:training log loss: 0.24363911746023934        \
        INFO:root:test log loss : 2.727398407356135             \
        INFO:root:runtime: 16.0 m') # deleting cp_type has little to no impact on the results and neither do the highly correlated features
