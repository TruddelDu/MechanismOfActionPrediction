import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  train_test_split, GridSearchCV#RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
import logging
import time
import joblib


# Random Forest has lower error values from the start than XGBoost but takes twice as long - 
# I'll continue with XGBoost due to time 

logging.basicConfig(level=logging.DEBUG)
logging.info(f'start @ {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
start = time.perf_counter()
## PREPROCESSING


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

joblib.dump(X_test, 'X_test.pickle', compress=('xz', 9))
joblib.dump(y_test, 'y_test.pickle', compress=('xz', 9))

## XGBoost MODEL
random_forest = RandomForestRegressor(max_depth=5, random_state=174)
pipe = Pipeline([
    ('rf_model', random_forest)
])

# params = {'rf_model__max_depth': [5, 10],
#             # 'rf_model__min_samples_split' : [2, 5],
#             # 'rf_model__min_samples_leaf' : [1, 2, 5]
# }

logging.info('Training model')
# search = GridSearchCV(pipe, params, cv=3, n_jobs=-2, verbose=2)
pipe.fit(X_train, y_train)

logging.info('Training model finished')


y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test)

logging.info(f'training log loss: {log_loss(y_train, y_train_pred)}')
logging.info(f'test log loss : {log_loss(y_test, y_test_pred)}')
end = time.perf_counter()
logging.info(f'runtime: {round((end-start)/60,1)} m')
# logging.info(f'best param: {search.best_params_}')



# max_depth = 10
# DEBUG:root:training log loss: 2.521059316604307
# DEBUG:root:test log loss : 2.703595872539878
# DEBUG:root:runtime: 32.1 m