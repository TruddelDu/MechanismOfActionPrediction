import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
import logging

## PREPROCESSING
logging.basicConfig(level=logging.DEBUG)

X_data = pd.read_csv('MechanismOfAction/train_features.csv')
y_data = pd.read_csv('MechanismOfAction/train_targets_scored.csv')

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.2, 
                                                    random_state=174)
logging.debug(f'Length of X_train: {len(X_train)}, {type(X_train)}')

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
logging.info('Training model')
reg.fit(X_train, y_train)
logging.info('Training model finished')

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

logging.debug(f'training log loss: {log_loss(y_train, y_train_pred)}')
logging.debug(f'test log loss : {log_loss(y_test, y_test_pred)}')


print('DEBUG:root:training log loss: 0.15360432849420547 \
DEBUG:root:test log loss : 14.039272039044583')