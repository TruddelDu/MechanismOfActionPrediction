import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
import logging
import time
import numpy as np

def reshape_prediction(pred):
    new_pred = np.zeros((len(pred), len(pred[0])))
    for i, n_class in enumerate(pred):
        if n_class.shape[1]<2:
            # in some cases none have been predicted of a particular class
            continue
        class_preds = [samp[1] for samp in n_class]
        new_pred[i] = class_preds
    return np.transpose(new_pred)

## configuration of logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler('iniModel_testing.log')
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

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.2, 
                                                    random_state=174)
logger.debug(f'Length of X_train: {len(X_train)}, {type(X_train)}')

# no missing values present in data

# save ids
id_train = X_train.sig_id.copy()
id_test  = X_test.sig_id.copy()

# Labelencoding cp_type, cp_dose
label_encoder = LabelEncoder()
for col in ['cp_type', 'cp_dose']:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])

# delete all unneeded columns
X_train.drop(columns=['sig_id'], inplace=True)
y_train.drop(columns=['sig_id'], inplace=True)
X_test.drop(columns=['sig_id'], inplace=True)
y_test.drop(columns=['sig_id'], inplace=True)

## ROUGH MODEL
reg = DecisionTreeClassifier(random_state=174)
logger.info('Training model')
reg.fit(X_train, y_train)
logger.info('Training model finished')

y_train_pred = reg.predict_proba(X_train)
y_test_pred = reg.predict_proba(X_test)

y_train_pred = reshape_prediction(y_train_pred)
y_test_pred = reshape_prediction(y_test_pred)

logger.info(f'training log loss: {log_loss(y_train, y_train_pred)}')
logger.info(f'test log loss : {log_loss(y_test, y_test_pred)}')
# logger.info(pipe.best_params_)
end = time.perf_counter()
logger.info(f'runtime: {round((end-start)/60,1)} m')

# 2021-12-02 17:36:26,694:INFO:training log loss: 0.15360432849421102
# 2021-12-02 17:36:26,826:INFO:test log loss : 14.039272039044585
# 2021-12-02 17:36:26,828:INFO:runtime: 142.2 m