import pandas as pd
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier 
#from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
import logging
import time


## configuration of logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler('XGB_testing.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info(f'start @ {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
start = time.perf_counter()

## PREPROCESSING
X_data = pd.read_csv('MechanismOfAction/train_features.csv')
y_data = pd.read_csv('MechanismOfAction/train_targets_scored.csv')

# delete all unneeded columns
X_data.drop(columns=['sig_id', 'cp_type'], inplace=True)
y_data.drop(columns=['sig_id'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.2, 
                                                    random_state=174)
logging.debug(f'Length of X_train: {len(X_train)}, {type(X_train)}')

# no missing values present in data

# Onehot encoding cp_type
one_hot = OneHotEncoder(drop='if_binary')

preproc_transformer = ColumnTransformer([
    ('onehot', one_hot, ['cp_dose'])], 
    remainder='passthrough')

## XGBoost MODEL
multioutputclassifier = OneVsRestClassifier(xgb.XGBClassifier(max_depth=5, tree_method='gpu_hist', eval_metric='logloss', use_label_encoder=False))

pipe = Pipeline([
    ('preproc', preproc_transformer),
    ('xgb_model', multioutputclassifier)
])

# params = {
#     # 'estimator__eta' : [0.2, 0.3, 0.5], #learning_rate
#     'xgb_model__estimator__max_depth' : [5, 10, 15],
# #    'estimator__scale_pos_weight' : [1, 0.5, 2],^
#     # 'estimator__max_delta_step' : [1]
# }



logger.info('Training model')
# search = GridSearchCV(pipe, params, verbose=2, cv=2, n_jobs=-2)
pipe.fit(X_train, y_train)

# for k,i in params.items():
#     logger.info(f'tested hyperparameters: {k}: {i}')

logging.info('Training model finished')


y_train_pred = pipe.predict_proba(X_train)
y_test_pred = pipe.predict_proba(X_test)

logger.info(f'training log loss: {log_loss(y_train, y_train_pred)}')
logger.info(f'test log loss : {log_loss(y_test, y_test_pred)}')
# logger.info(pipe.best_params_)
end = time.perf_counter()
logger.info(f'runtime: {round((end-start)/60,1)} m')


# INFO:__main__:training log loss: 0.16613967350478034
# INFO:__main__:test log loss : 2.775775177029787
# INFO:__main__:runtime: 19.8 m