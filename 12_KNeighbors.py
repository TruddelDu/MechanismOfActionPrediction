# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:38:04 2021

@author: angel
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

file_handler = logging.FileHandler('knn_testing.log')
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

scaler = StandardScaler()

## MODEL
model = KNeighborsClassifier()

pipe = Pipeline([
    ('preproc', preproc_transformer),
    ('scaler', scaler),
    # ('model', model)
])

X_train = pipe.fit_transform(X_train)
X_test = pipe.transform(X_test)

params = {
    'n_neighbors' : range(5,50,5),
    'weights' : ['uniform', 'distance'],
    'leaf_size' : range(15,46,15),
    'p' : [1,2],
}

logger.info('Training model')
search = GridSearchCV(model, params,cv=3, verbose=3, n_jobs=-2)
search.fit(X_train, y_train)

for k,i in params.items():
    logger.info(f'tested hyperparameters: {k}: {i}')

logging.info('Training model finished')


y_train_pred = search.predict_proba(X_train)
y_test_pred = search.predict_proba(X_test)


y_train_pred = reshape_prediction(y_train_pred)
y_test_pred = reshape_prediction(y_test_pred)

logger.info(f'training log loss: {log_loss(y_train, y_train_pred)}')
logger.info(f'test log loss : {log_loss(y_test, y_test_pred)}')
logger.info(search.best_params_)
end = time.perf_counter()
logger.info(f'runtime: {round((end-start)/60,1)} m')

# simple model after standard scaling
# INFO:__main__:training log loss: 0.775718522386592
# INFO:__main__:test log loss : 17.264556869142318
# INFO:__main__:runtime: 0.5 m

# parameter optimized
# INFO:__main__:Training model
# Fitting 3 folds for each of 108 candidates, totalling 324 fits
# INFO:__main__:tested hyperparameters: n_neighbors: range(5, 50, 5)
# INFO:__main__:tested hyperparameters: weights: ['uniform', 'distance']
# INFO:__main__:tested hyperparameters: leaf_size: range(15, 46, 15)
# INFO:__main__:tested hyperparameters: p: [1, 2]
# INFO:__main__:training log loss: 0.775718522386592
# INFO:__main__:test log loss : 17.264556869142318
# INFO:__main__:{'leaf_size': 15, 'n_neighbors': 5, 'p': 2, 'weights': 'uniform'}
# INFO:__main__:runtime: 98.5 m