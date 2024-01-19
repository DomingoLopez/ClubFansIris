from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTEN
import pandas as pd
import numpy as np

def encoder(df_train, df_test):
    orden_x24 = ["VLOW", "LOW", "MED", "HIGH", "VHIGH"]

    ordinal_encoder_x24 = OrdinalEncoder(categories=[orden_x24], dtype=int)

    df_train["X24"] = ordinal_encoder_x24.fit_transform(df_train[["X24"]])
    df_test["X24"] = ordinal_encoder_x24.transform(df_test[["X24"]])

    orden_x25 = ["NO", "YES"]

    ordinal_encoder_x25 = OrdinalEncoder(categories=[orden_x25], dtype=int)

    df_train["X25"] = ordinal_encoder_x25.fit_transform(df_train[["X25"]])
    df_test["X25"] = ordinal_encoder_x25.transform(df_test[["X25"]])

    df_train_encoded = df_train.copy()
    df_test_encoded = df_test.copy()

    df_train_encoded.loc[df_train["X30"] == "VTKGN", "X30"] = 1
    df_train_encoded.loc[df_train["X30"] != "VTKGN", "X30"] = 0

    df_test_encoded.loc[df_test["X30"] == "VTKGN", "X30"] = 1
    df_test_encoded.loc[df_test["X30"] != "VTKGN", "X30"] = 0

    df_train_encoded["X30"] = pd.to_numeric(df_train_encoded["X30"])
    df_test_encoded["X30"] = pd.to_numeric(df_train_encoded["X30"])

    return df_train_encoded, df_test_encoded

def train_validation_split(df_train):
    df_train_label = df_train["RATE"].to_frame()
    df_train = df_train.drop(["RATE"], axis=1, inplace=False)

    (
        df_train,
        df_validation,
        df_train_label,
        df_validation_label,
    ) = train_test_split(
        df_train,
        df_train_label,
        test_size=int(len(df_train) * 0.3),
        random_state=42,
        stratify=df_train_label,
    )

    df_train = df_train.join(other=df_train_label).sort_index().reset_index(drop=True)
    df_validation = (
        df_validation.join(other=df_validation_label)
        .sort_index()
        .reset_index(drop=True)
    )

    return df_train, df_validation

def train_resampling(df_train):
    train_id = df_train["ID"].to_frame()
    train_labels = df_train["RATE"].to_frame()
    train_data = df_train.drop(["RATE", "ID"], axis=1, inplace=False)

    sm = SMOTEN(random_state=42)

    X_res, y_res = sm.fit_resample(train_data, train_labels)

    train_id_dict = train_id.to_dict()
    key = len(train_id)
    for i in np.arange(9000, 9000 + (len(X_res) - len(train_data))):
        train_id_dict["ID"][key] = i
        key += 1

    train_id = pd.DataFrame.from_dict(train_id_dict)

    df_train = train_id.join(other=[X_res, y_res])
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_train

def outliers_remove(df_train):
    clf = LocalOutlierFactor()
    train_id = df_train["ID"].to_frame()
    train_labels = df_train["RATE"].to_frame()
    train_data = df_train.drop(["RATE", "ID"], axis=1, inplace=False)

    # print(pd.isnull(train_data).any(1).nonzero()[0])
    
    outliers_index = np.argwhere(clf.fit_predict(train_data) == 1).flatten()

    # print(len(outliers_index))
    # print(np.argwhere(clf.fit_predict(train_data) == -1).flatten())

    train_id = train_id.iloc[outliers_index]
    train_labels = train_labels.iloc[outliers_index]
    train_data = train_data.iloc[outliers_index]

    df_train = train_id.join(other=[train_data, train_labels])
    df_train = df_train.reset_index(drop=True)

    # print(len(train_id))
    # print(len(train_labels))
    # print(len(train_data))
    # print(df_train)

    return df_train

def scale(df_train, df_validation, df_test, scaler):
    train_id = df_train["ID"].to_frame()
    train_labels = df_train["RATE"].to_frame()
    train_data = df_train.drop(["RATE", "ID"], axis=1, inplace=False)

    scaler.fit(train_data)
    scale_X_train = pd.DataFrame(
        scaler.transform(train_data), columns=train_data.columns
    )

    df_train = train_id.join(other=[scale_X_train, train_labels])

    validation_id = df_validation["ID"].to_frame()
    validation_labels = df_validation["RATE"].to_frame()
    validation_data = df_validation.drop(["RATE", "ID"], axis=1, inplace=False)

    scale_X_validation = pd.DataFrame(
        scaler.transform(validation_data), columns=train_data.columns
    )

    df_validation = validation_id.join(other=[scale_X_validation, validation_labels])

    test_id = df_test["ID"].to_frame()
    test_data = df_test.drop(["ID"], axis=1, inplace=False)

    scale_X_test = pd.DataFrame(
        scaler.transform(test_data), columns=test_data.columns
    )

    df_test = test_id.join(other=scale_X_test)

    return df_train, df_validation, df_test

def imputation(df_train, df_validation, df_test, imputer_func):
    train_id = df_train["ID"].to_frame()
    train_labels = df_train["RATE"].to_frame()
    train_data = df_train.drop(["RATE", "ID"], axis=1, inplace=False)

    imp_train = imputer_func.fit(train_data)
    imputed_X_train = pd.DataFrame(
        imp_train.transform(train_data), columns=train_data.columns
    )

    df_train = train_id.join(other=[imputed_X_train, train_labels])

    validation_id = df_validation["ID"].to_frame()
    validation_labels = df_validation["RATE"].to_frame()
    validation_data = df_validation.drop(["RATE", "ID"], axis=1, inplace=False)

    imputed_X_validation = pd.DataFrame(
        imp_train.transform(validation_data), columns=validation_data.columns
    )

    df_validation = validation_id.join(other=[imputed_X_validation, validation_labels])

    test_id = df_test["ID"].to_frame()
    test_data = df_test.drop(["ID"], axis=1, inplace=False)

    imputed_X_test = pd.DataFrame(
        imp_train.transform(test_data), columns=test_data.columns
    )

    df_test = test_id.join(other=imputed_X_test)

    return df_train, df_validation, df_test

def apply_PCA(df_train, df_validation, df_test, n_components):
    train_id = df_train["ID"].to_frame()
    train_labels = df_train["RATE"].to_frame()
    train_data = df_train.drop(["RATE", "ID"], axis=1, inplace=False)

    pca = PCA(n_components=n_components)
    pca_train = pca.fit(train_data)

    selection_train = pd.DataFrame(
        pca_train.transform(train_data),
        columns=pca_train.get_feature_names_out(train_data.columns),
    )

    n_features = len(pca_train.get_feature_names_out(train_data.columns))

    df_train = train_id.join(other=[selection_train, train_labels])

    validation_id = df_validation["ID"].to_frame()
    validation_labels = df_validation["RATE"].to_frame()
    validation_data = df_validation.drop(["RATE", "ID"], axis=1, inplace=False)

    selection_validation = pd.DataFrame(
        pca_train.transform(validation_data),
        columns=pca_train.get_feature_names_out(validation_data.columns),
    )

    df_validation = validation_id.join(other=[selection_validation, validation_labels])

    test_id = df_test["ID"].to_frame()
    test_data = df_test.drop(["ID"], axis=1, inplace=False)

    selection_test = pd.DataFrame(
        pca_train.transform(test_data),
        columns=pca_train.get_feature_names_out(test_data.columns),
    )

    df_test = test_id.join(other=selection_test)

    return df_train, df_validation, df_test, n_features

def feature_selection(df_train, df_validation, df_test, feature_selector):
    train_id = df_train["ID"].to_frame()
    train_labels = df_train["RATE"].to_frame()
    train_data = df_train.drop(["RATE", "ID"], axis=1, inplace=False)

    feature_train = feature_selector.fit(train_data, train_labels.to_numpy().flatten())

    selection_train = pd.DataFrame(
        feature_train.transform(train_data),
        columns=feature_train.get_feature_names_out(train_data.columns),
    )

    n_features = len(feature_train.get_feature_names_out(train_data.columns))

    df_train = train_id.join(other=[selection_train, train_labels])

    validation_id = df_validation["ID"].to_frame()
    validation_labels = df_validation["RATE"].to_frame()
    validation_data = df_validation.drop(["RATE", "ID"], axis=1, inplace=False)

    selection_validation = pd.DataFrame(
        feature_train.transform(validation_data),
        columns=feature_train.get_feature_names_out(validation_data.columns),
    )

    df_validation = validation_id.join(other=[selection_validation, validation_labels])

    test_id = df_test["ID"].to_frame()
    test_data = df_test.drop(["ID"], axis=1, inplace=False)

    selection_test = pd.DataFrame(
        feature_train.transform(test_data),
        columns=feature_train.get_feature_names_out(test_data.columns),
    )

    df_test = test_id.join(other=selection_test)

    return df_train, df_validation, df_test, n_features

def scale_test(df_train, df_test, scaler):
    train_id = df_train["ID"].to_frame()
    train_labels = df_train["RATE"].to_frame()
    train_data = df_train.drop(["RATE", "ID"], axis=1, inplace=False)

    scaler.fit(train_data)
    scale_X_train = pd.DataFrame(
        scaler.transform(train_data), columns=train_data.columns
    )

    df_train = train_id.join(other=[scale_X_train, train_labels])

    test_id = df_test["ID"].to_frame()
    test_data = df_test.drop(["ID"], axis=1, inplace=False)

    scale_X_test = pd.DataFrame(
        scaler.transform(test_data), columns=test_data.columns
    )

    df_test = test_id.join(other=scale_X_test)

    return df_train, df_test

def imputation_test(df_train, df_test, imputer_func):
    train_id = df_train["ID"].to_frame()
    train_labels = df_train["RATE"].to_frame()
    train_data = df_train.drop(["RATE", "ID"], axis=1, inplace=False)

    imp_train = imputer_func.fit(train_data)
    imputed_X_train = pd.DataFrame(
        imp_train.transform(train_data), columns=train_data.columns
    )

    df_train = train_id.join(other=[imputed_X_train, train_labels])

    test_id = df_test["ID"].to_frame()
    test_data = df_test.drop(["ID"], axis=1, inplace=False)

    imputed_X_test = pd.DataFrame(
        imp_train.transform(test_data), columns=test_data.columns
    )

    df_test = test_id.join(other=imputed_X_test)

    return df_train, df_test

def apply_PCA_test(df_train, df_test, n_components):
    train_id = df_train["ID"].to_frame()
    train_labels = df_train["RATE"].to_frame()
    train_data = df_train.drop(["RATE", "ID"], axis=1, inplace=False)

    pca = PCA(n_components=n_components)
    pca_train = pca.fit(train_data)

    selection_train = pd.DataFrame(
        pca_train.transform(train_data),
        columns=pca_train.get_feature_names_out(train_data.columns),
    )

    n_features = len(pca_train.get_feature_names_out(train_data.columns))

    df_train = train_id.join(other=[selection_train, train_labels])

    test_id = df_test["ID"].to_frame()
    test_data = df_test.drop(["ID"], axis=1, inplace=False)

    selection_test = pd.DataFrame(
        pca_train.transform(test_data),
        columns=pca_train.get_feature_names_out(test_data.columns),
    )

    df_test = test_id.join(other=selection_test)

    return df_train, df_test, n_features

def feature_selection_test(df_train, df_test, feature_selector):
    train_id = df_train["ID"].to_frame()
    train_labels = df_train["RATE"].to_frame()
    train_data = df_train.drop(["RATE", "ID"], axis=1, inplace=False)

    feature_train = feature_selector.fit(train_data, train_labels.to_numpy().flatten())

    selection_train = pd.DataFrame(
        feature_train.transform(train_data),
        columns=feature_train.get_feature_names_out(train_data.columns),
    )

    n_features = len(feature_train.get_feature_names_out(train_data.columns))

    df_train = train_id.join(other=[selection_train, train_labels])

    test_id = df_test["ID"].to_frame()
    test_data = df_test.drop(["ID"], axis=1, inplace=False)

    selection_test = pd.DataFrame(
        feature_train.transform(test_data),
        columns=feature_train.get_feature_names_out(test_data.columns),
    )

    df_test = test_id.join(other=selection_test)

    return df_train, df_test, n_features
