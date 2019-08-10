"""
Authors:
        ITryagain <long452a@163.com>

Reference:
        https://www.kaggle.com/vprokopev/mean-likelihood-encodings-a-comprehensive-study

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def one_hot_encode(train_data, test_data, columns):
    """
    :return: Returns the train and test DataFrame with encoded columns
    """
    all_data = pd.concat([train_data, test_data], axis=0)
    encoded_cols = []
    for col in columns:
        encoded_cols.append(pd.get_dummies(all_data[col], prefix='one_hot_'+col, drop_first=True))
    all_encoded = pd.concat(encoded_cols, axis=1)
    return all_encoded.iloc[:train_data.shape[0], :], all_encoded.iloc[train_data.shape[0]:, :]


def label_encode(train_data, test_data, columns):
    """
    :return: Returns the train and test DataFrame with encoded columns
    """
    encoded_cols = []
    for col in columns:
        factorised = pd.factorize(train_data[col])[1]
        labels = pd.Series(range(len(factorised)), index=factorised)
        encoded_train = train_data[col].map(labels)
        encode_test = test_data[col].map(labels)
        encoded_col = pd.concat([encoded_train, encode_test], axis=0)
        encoded_col[encoded_col.isnull()] = -1
        encoded_cols.append(pd.DataFrame({'label_'+col:encoded_col}))
    all_encoded = pd.concat(encoded_cols, axis=1)
    return all_encoded.iloc[:train_data.shape[0], :], all_encoded.iloc[train_data.shape[0]:, :]


def freq_encode(train_data, test_data, columns):
    """
    :return: Returns the train and test DataFrame with encoded columns
    """
    encoded_cols = []
    nsamples = train_data.shape[0]
    for col in columns:
        freqs_cat = train_data.groupby(col)[col].count() / nsamples
        encoded_train = train_data[col].map(freqs_cat)
        encoded_test = test_data[col].map(freqs_cat)
        encoded_col = pd.concat([encoded_train, encoded_test], axis=0)
        encoded_col[encoded_col.isnull()] = 0
        encoded_cols.append(pd.DataFrame({'freq_'+col:encoded_col}))
    all_encoded = pd.concat(encoded_cols, axis=1)
    return all_encoded.iloc[:train_data.shape[0], :], all_encoded.iloc[train_data.shape[0]:, :]


def mean_encode(train_data, test_data, columns, target_col, reg_method=None,
                alpha=0, add_random=False, rmean=0, rstd=0.1, folds=1):
    """
    :return: Returns the train and test DataFrame with encoded columns
    """
    encoded_cols = []
    target_mean_global = train_data[target_col].mean()
    for col in columns:
        # Getting means for test data
        nrows_cat = train_data.groupby(col)[target_col].count()
        target_means_cats = train_data.groupby(col)[target_col].mean()
        target_means_cats_adj = (target_means_cats*nrows_cat +
                                 target_mean_global*alpha)/(nrows_cat+alpha)
        # Mapping means to test data
        encoded_test = test_data[col].map(target_means_cats_adj)
        # Getting a train encodings
        if reg_method == 'expanding_mean':
            train_data_shuffled = train_data.sample(frac=1, random_state=1)
            cumsum = train_data_shuffled.groupby(col)[target_col].cumsum() - train_data_shuffled[target_col]
            cumcnt = train_data_shuffled.groupby(col).cumcount()
            encoded_train = cumsum / cumcnt
            encoded_train.fillna(target_mean_global, inplace=True)
            if add_random:
                encoded_train = encoded_train + np.random.normal(loc=rmean, scale=rstd,
                                                               size=(encoded_train.shape[0]))
        elif (reg_method == 'k_fold') and (folds > 1):
            kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2319)
            parts = []
            for tr_in, val_ind in enumerate(folds.split(train_data.values, train_data.values)):
                # divide data
                df_for_estimation, df_estimated = train_data.iloc[tr_in], train_data.iloc[val_ind]
                # getting means on data for estimation (all folds except estimated)
                nrows_cat = df_for_estimation.groupby(col)[target_col].count()
                target_means_cats = df_for_estimation.groupby(col)[target_col].mean()
                target_means_cats_adj = (target_means_cats*nrows_cat +
                                         target_mean_global*alpha)/(nrows_cat+alpha)
                # Mapping means to estimated fold
                encoded_train_part = df_estimated[col].map(target_means_cats_adj)
                if add_random:
                    encoded_train_part = encoded_train_part + np.random.normal(loc=rmean, scale=rstd,
                                                                             size=(encoded_train_part.shape[0]))
                # Saving estimated encodings for a fold
                parts.append(encoded_train_part)
            encoded_train = pd.concat(parts, axis=0)
            encoded_train.fillna(target_mean_global, inplace=True)
        else:
            encoded_train = train_data[col].map(target_means_cats_adj)
            if add_random:
                encoded_train = encoded_train + np.random.normal(loc=rmean, scale=rstd,
                                                               size=(encoded_train.shape[0]))

        # Saving the column with means
        encoded_col = pd.concat([encoded_train, encoded_test], axis=0)
        encoded_col[encoded_col.isnull()] = target_mean_global
        encoded_cols.append(pd.DataFrame({'mean_'+target_col+'_'+col: encoded_col}))
    all_encoded = pd.concat(encoded_cols, axis=1)
    return all_encoded.iloc[:train_data.shape[0], :], all_encoded.iloc[train_data.shape[0]:, :]
