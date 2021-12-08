import pandas as pd
# from dask.distributed import Client, LocalCluster
# from dask_ml.model_selection import HyperbandSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
import logging
import time


## configuration of logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler('hyperparamter_search.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info(f'start @ {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
start = time.perf_counter()


# client = Client()
## PREPROCESSING
X_data = pd.read_csv('MechanismOfAction/train_features.csv')
y_data = pd.read_csv('MechanismOfAction/train_targets_scored.csv')

# delete all unneeded columns
X_data.drop(columns=['sig_id', 'cp_type'], inplace=True) 
y_data.drop(columns=['sig_id'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.2, 
                                                    random_state=174)
logger.debug(f'Length of X_train: {len(X_train)}, {type(X_train)}')

# no missing values present in data

# Onehot encoding cp_type
one_hot = OneHotEncoder(drop='if_binary')

# PCA
genes = [col for col in X_train.columns if col[:2]=='g-']
cell_lines = [col for col in X_train.columns if col[:2]=='c-']

pca_genes = PCA(random_state=174, n_components=50) # n_components=50
pca_cells = PCA(random_state=174, n_components=15)  # n_components=15

preproc_transformer = ColumnTransformer([
    ('onehot', one_hot, ['cp_dose']), 
    ('pca_genes', pca_genes, genes),#[list(X_train.columns).index(col) for col in genes]),
    ('pca_cells', pca_cells, cell_lines)],#[list(X_train.columns).index(col) for col in cell_lines])],
    remainder='passthrough')

## XGBoost MODEL
multioutputclassifier = MultiOutputClassifier(xgb.XGBClassifier(tree_method='gpu_hist', eval_metric='logloss'))

pipe = Pipeline([
    ('preproc', preproc_transformer),
    ('xgb_model', multioutputclassifier)
])

# params = {'preproc__pca_cells__n_components':  range(5,26,5) , #100 original cell features --> range(5,26,5)
#           'preproc__pca_genes__n_components' : range(40,201,40), #772 original gene features --> range(40,201,40)
#           'xgb_model__estimator__max_depth' : [3, 5, 7],
#           'xgb_model__estimator__min_child_weight' : [1,2,50],
#           'xgb_model__estimator__scale_pos_weight' : [300, 600, 1200], # should be sum(neg instances)/sum(pos instances). Dependent on the output thats 1180+/-2483. median: 618
# }

params = {'xgb_model__estimator__max_depth' : [5],
}


logger.info('Training model')
search = GridSearchCV(pipe, params, scoring='neg_log_loss', verbose=3)#, max_iter=21)#, patience=True, scoring='neg_log_loss', verbose=True)
search.fit(X_train, y_train)

for k,i in params.items():
    logger.info(f'tested hyperparameters: {k}: {i}')

logger.info('Training model finished')


y_train_pred = search.predict_proba(X_train)
y_test_pred = search.predict_proba(X_test)

logger.info(f'training log loss: {log_loss(y_train, y_train_pred)}')
logger.info(f'test log loss : {log_loss(y_test, y_test_pred)}')
logger.info(f'{search.best_params_}')
logger.info(f'{search.cv.results_}')
end = time.perf_counter()
logger.info(f'runtime: {round((end-start)/60,1)} m')

