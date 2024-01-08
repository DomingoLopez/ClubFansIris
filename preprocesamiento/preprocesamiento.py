import numpy as np
import pandas as pd
import os as os

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer

# Cargamos csv con los datos de train
df_train = pd.read_csv("../data_raw/training_data.csv", sep=",", header=0, na_values=['?', '', 'NA'])
# Cargamos csv con los datos de test
df_test = pd.read_csv("../data_raw/test_data.csv", sep=",", header=0, na_values=['?', '', 'NA'])

df_train_num = df_train.copy()
df_test_num = df_test.copy()

# 1. "OrdinalEncoder" para X24
orden_x24 = ['VLOW', 'LOW', 'MED', 'HIGH', 'VHIGH']

ordinal_encoder_x24 = OrdinalEncoder(categories=[orden_x24], dtype=int)

df_train_num['X24'] = ordinal_encoder_x24.fit_transform(df_train_num[['X24']])
df_test_num['X24'] = ordinal_encoder_x24.transform(df_test_num[['X24']])

# 2. "OrdinalEncoder" para X25
orden_x25 = ['NO', 'YES']

ordinal_encoder_x25 = OrdinalEncoder(categories=[orden_x25], dtype=int)

df_train_num['X25'] = ordinal_encoder_x25.fit_transform(df_train_num[['X25']])
df_test_num['X25'] = ordinal_encoder_x25.transform(df_test_num[['X25']])

# Si es VTKGN 1 else 0
# Ya que la la clase está muy desbalanceada
df_train_encoded = df_train_num.copy()
df_test_encoded = df_test_num.copy()

df_train_encoded.loc[df_train_num['X30'] == 'VTKGN', 'X30'] = 1
df_train_encoded.loc[df_train_num['X30'] != 'VTKGN', 'X30'] = 0

df_test_encoded.loc[df_test_num['X30'] == 'VTKGN', 'X30'] = 1
df_test_encoded.loc[df_test_num['X30'] != 'VTKGN', 'X30'] = 0

df_train_encoded['X30'] = pd.to_numeric(df_train_encoded['X30'])
df_test_encoded['X30'] = pd.to_numeric(df_train_encoded['X30']) 

# ### Imputación de Missing Values

# Imputando con KNN (TRAIN)

train_id = df_train_encoded["ID"]
train_labels = df_train_encoded['RATE']
train_data = df_train_encoded.drop(['RATE', 'ID'], axis=1, inplace=False)

Knn_imp_train = KNNImputer(n_neighbors=4).fit(train_data)
imputed_X_train = pd.DataFrame(Knn_imp_train.transform(train_data), columns=train_data.columns)

result_df_train = train_id.to_frame().join(other=[imputed_X_train, train_labels])

# Imputando con KNN (TEST)

test_id = df_test_encoded["ID"]
test_data = df_test_encoded.drop(['ID'], axis=1, inplace=False)

imputed_X_test = pd.DataFrame(Knn_imp_train.transform(test_data), columns=test_data.columns)

result_df_test = test_id.to_frame().join(other=imputed_X_test)

# ### Exportación a carpeta de Preprocesamiento

# Conversión de aquellas que eran enteros a enteros tras imputación knn
result_df_train['ID'] = result_df_train['ID'].astype('int')
result_df_train['X1'] = result_df_train['X1'].astype('int')
result_df_train['X2'] = result_df_train['X2'].astype('int')
result_df_train['X3'] = result_df_train['X3'].astype('int')
result_df_train['X5'] = result_df_train['X5'].astype('int')
result_df_train['X7'] = result_df_train['X7'].astype('int')
result_df_train['X24'] = result_df_train['X24'].astype('int')
result_df_train['X25'] = result_df_train['X25'].astype('int')
result_df_train['X30'] = result_df_train['X30'].astype('int')

result_df_test['ID'] = result_df_test['ID'].astype('int')
result_df_test['X1'] = result_df_test['X1'].astype('int')
result_df_test['X2'] = result_df_test['X2'].astype('int')
result_df_test['X3'] = result_df_test['X3'].astype('int')
result_df_test['X5'] = result_df_test['X5'].astype('int')
result_df_test['X7'] = result_df_test['X7'].astype('int')
result_df_test['X24'] = result_df_test['X24'].astype('int')
result_df_test['X25'] = result_df_test['X25'].astype('int')
result_df_test['X30'] = result_df_test['X30'].astype('int')

result_df_train.to_csv('../data_preprocess/train_preprocess.csv', index=False)
result_df_test.to_csv('../data_preprocess/test_preprocess.csv', index=False)