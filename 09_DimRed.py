import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import logging
import time


logging.basicConfig(level=logging.DEBUG)
logging.info(f'start @ {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
start = time.perf_counter()

X_data = pd.read_csv('MechanismOfAction/train_features.csv')
y_data = pd.read_csv('MechanismOfAction/train_targets_scored.csv')

# delete all unneeded columns
X_data.drop(columns=['sig_id', 'cp_type'], inplace=True) #, 'c-42', 'c-4', 'c-13', 'c-94', 'c-2', 'c-31'
y_data.drop(columns=['sig_id'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.2, 
                                                    random_state=174)


# Labelencoding cp_dose
label_encoder = LabelEncoder()

# PCA
genes = [col for col in X_train.columns if col[:2]=='g-']
cell_lines = [col for col in X_train.columns if col[:2]=='c-']

pca_genes = PCA(random_state=174) # n_components=50
pca_cells = PCA(random_state=174)  # n_components=15

categ_encode = ColumnTransformer([
    ('label_enc', label_encoder, ['cp_dose'])], 
    ('pca_genes', pca_genes),
    ('pca_genes', pca_cells),
    remainder='passthrough')


random_forest = RandomForestRegressor( random_state=174)

pipe = Pipeline([
    ('preproc', categ_encode),
    ('rf_model', random_forest)
])

logging.info(sorted(pipe.get_params().keys()))

params = {'rf_model__max_depth': [5, 10, 15],
            'rf_model__min_samples_split' : [2, 5, 8],
            'rf_model__min_samples_leaf' : [1, 2, 5]
}

search = RandomizedSearchCV(pipe, params,cv=3, scoring='neg_log_loss', n_jobs=-2, verbose=2)

logging.info('Training model')
search.fit(X_train, y_train)

logging.info('Training model finished')


y_train_pred = search.predict(X_train)
y_test_pred = search.predict(X_test)

logging.info(f'training log loss: {log_loss(y_train, y_train_pred)}')
logging.info(f'test log loss : {log_loss(y_test, y_test_pred)}')
end = time.perf_counter()
logging.info(f'runtime: {round((end-start)/60,1)} m')
logging.info(f'best param: {search.best_params_}')