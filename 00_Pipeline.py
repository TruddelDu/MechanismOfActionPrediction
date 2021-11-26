import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss
import logging
import time

start = time.perf_counter()

## PREPROCESSING
logging.basicConfig(level=logging.DEBUG)
logging.info(f'start @ {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')

X_data = pd.read_csv('MechanismOfAction/train_features.csv')
y_data = pd.read_csv('MechanismOfAction/train_targets_scored.csv')

# delete all unneeded columns
X_data.drop(columns=['sig_id', 'cp_type', 'c-42', 'c-4', 'c-13', 'c-94', 'c-2', 'c-31'], inplace=True) #, 'c-42', 'c-4', 'c-13', 'c-94', 'c-2', 'c-31'
y_data.drop(columns=['sig_id'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.2, 
                                                    random_state=174)

# no missing values present in data

# Onehot encoding cp_type
one_hot = OneHotEncoder(drop='if_binary')

categ_encode = ColumnTransformer([
    ('onehot', one_hot, ['cp_dose'])],
    remainder='passthrough')

## XGBoost MODEL

random_forest = RandomForestRegressor(max_depth=10, random_state=174)

pipe = Pipeline([
    ('preproc', categ_encode),
    ('rf_model', random_forest)
])

params = {'rf_model__max_depth': [15],
#            'rf_model__min_samples_split' : [2, 5, 8],
#            'rf_model__min_samples_leaf' : [1, 2, 5]
}


#search = RandomizedSearchCV(pipe, params,cv=3, scoring='neg_log_loss', n_jobs=-2, verbose=2)

logging.info('Training model')
pipe.fit(X_train, y_train)

logging.info('Training model finished')


y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test)

logging.info(f'training log loss: {log_loss(y_train, y_train_pred)}')
logging.info(f'test log loss : {log_loss(y_test, y_test_pred)}')
end = time.perf_counter()
logging.info(f'runtime: {round((end-start)/60,1)} m')
#logging.info(f'best param: {search.best_params_}')



# max_depth = 10
# DEBUG:root:training log loss: 2.521059316604307
# DEBUG:root:test log loss : 2.703595872539878
# DEBUG:root:runtime: 32.1 m