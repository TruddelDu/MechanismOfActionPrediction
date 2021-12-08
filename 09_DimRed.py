import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV # RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import logging
import time

## configuration of logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler('DimRed_testing.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info(f'start @ {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
start = time.perf_counter()


# Start model

X_data = pd.read_csv('MechanismOfAction/train_features.csv')
y_data = pd.read_csv('MechanismOfAction/train_targets_scored.csv')

# delete all unneeded columns
X_data.drop(columns=['sig_id', 'cp_type'], inplace=True) #, 'c-42', 'c-4', 'c-13', 'c-94', 'c-2', 'c-31'
y_data.drop(columns=['sig_id'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.2, 
                                                    random_state=174)


# Onehot encoding cp_type
one_hot = OneHotEncoder(drop='if_binary')

# PCA
genes = [col for col in X_train.columns if col[:2]=='g-']
cell_lines = [col for col in X_train.columns if col[:2]=='c-']

pca_genes = PCA(random_state=174) # n_components=50
pca_cells = PCA(random_state=174)  # n_components=15

preproc_transformer = ColumnTransformer([
    ('onehot', one_hot, ['cp_dose']), 
    # ('pca_genes', pca_genes, genes),#[list(X_train.columns).index(col) for col in genes]),
    ('pca_cells', pca_cells, cell_lines)],#[list(X_train.columns).index(col) for col in cell_lines])],
    remainder='passthrough')


## XGBoost MODEL
multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(tree_method='gpu_hist', eval_metric='logloss'))

pipe = Pipeline([
    ('preproc', preproc_transformer),
    ('xgb model', multioutputregressor)
])


params = {'preproc__pca_cells__n_components':  range(5,26,5) , #100 original cell features --> range(5,26,5)
            # 'preproc__pca_genes__n_components' : range(40,201,40), #772 original gene features --> range(40,201,40)
            
}

cv_splits = 2
search = GridSearchCV(pipe, params, cv=cv_splits, n_jobs=-2, verbose=2)

logger.info(f'GridSearch CV splits = {cv_splits}')

logger.info('Training model')
search.fit(X_train, y_train)

logger.info('Training model finished')


for k,i in params.items():
    logger.info(f'tested hyperparameters: {k}: {i}')


y_train_pred = search.predict(X_train)
y_test_pred = search.predict(X_test)

logger.info(f'training log loss: {log_loss(y_train, y_train_pred)}')
logger.info(f'test log loss : {log_loss(y_test, y_test_pred)}')
end = time.perf_counter()
logger.info(f'runtime: {round((end-start)/60,1)} m')
logger.info(f'best param: {search.best_params_}')