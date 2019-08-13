"""
Authors:
        ITryagain <long452a@163.com>

Reference:
        https://www.kaggle.com/vprokopev/mean-likelihood-encodings-a-comprehensive-study
        https://www.kaggle.com/c/avito-demand-prediction/discussion/60059 (Bayesian target encoding)
        https://www.kaggle.com/tnarik/likelihood-encoding-of-categorical-features (likelihood encoding)
        https://zhuanlan.zhihu.com/p/40231966
        https://mp.weixin.qq.com/s/U93vvFwZ8vSJuswk24yc6w

"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold


def one_hot_encode(train_data, test_data, columns):
    """
    :return: Returns the train and test DataFrame with encoded columns
    """
    all_data = pd.concat([train_data, test_data], axis=0)
    encoded_cols = []
    with tqdm(range(len(columns)), 'one_hot_encoding...') as t:
        for i in t:
            try:
                col = columns[i]
                encoded_cols.append(pd.get_dummies(all_data[col], prefix='one_hot_'+col))
            except StopIteration:
                break
    all_encoded = pd.concat(encoded_cols, axis=1)
    return all_encoded.iloc[:train_data.shape[0], :], all_encoded.iloc[train_data.shape[0]:, :]


def label_encode(train_data, test_data, columns, is_concat=False, fill_na=True, sort=True):
    """
    :param: fill_na: if fill_na == True will fill nan with -1, default True
    :param: is_concat: encoding时是否合并train_data和test_data
    :param: sort: 映射时是否需要排序
    :return: Returns the train and test DataFrame with encoded columns
    """
    encoded_cols = []
    if is_concat:
        all_data = pd.concat([train_data, test_data], axis=0)
        with tqdm(range(len(columns)), 'label_encoding...') as t:
            for i in t:
                try:
                    col = columns[i]
                    factorised = pd.factorize(all_data[col], sort=sort)[1]
                    labels = pd.Series(range(len(factorised)), index=factorised)
                    encoded_col = all_data[col].map(labels)
                    if fill_na:
                        encoded_col[encoded_col.isnull()] = -1
                    encoded_cols.append(pd.DataFrame({'label_'+col: encoded_col}))
                except StopIteration:
                    break
    else:
        with tqdm(range(len(columns)), 'label_encoding...') as t:
            for i in t:
                try:
                    col = columns[i]
                    factorised = pd.factorize(train_data[col], sort=sort)[1]
                    labels = pd.Series(range(len(factorised)), index=factorised)
                    encoded_train = train_data[col].map(labels)
                    encode_test = test_data[col].map(labels)
                    encoded_col = pd.concat([encoded_train, encode_test], axis=0)
                    if fill_na:
                        encoded_col[encoded_col.isnull()] = -1
                    encoded_cols.append(pd.DataFrame({'label_'+col: encoded_col}))
                except StopIteration:
                    break
    all_encoded = pd.concat(encoded_cols, axis=1)
    return all_encoded.iloc[:train_data.shape[0], :], all_encoded.iloc[train_data.shape[0]:, :]


def freq_encode(train_data, test_data, columns, is_concat=False, fill_na=True):
    """
    :param: fill_na: if fill_na == True will fill nan with 0, default True
    :param: is_concat: encoding时是否合并train_data和test_data
    :return: Returns the train and test DataFrame with encoded columns
    """
    encoded_cols = []
    nsamples = train_data.shape[0]
    if is_concat:
        all_data = pd.concat([train_data, test_data], axis=0)
        with tqdm(range(len(columns)), 'freq_encoding...') as t:
            for i in t:
                try:
                    col = columns[i]
                    freqs_cat = all_data.groupby(col)[col].count() / nsamples
                    encoded_col = all_data[col].map(freqs_cat)
                    if fill_na:
                        encoded_col[encoded_col.isnull()] = 0
                    encoded_cols.append(pd.DataFrame({'freq_'+col: encoded_col}))
                except StopIteration:
                    break
    else:
        with tqdm(range(len(columns)), 'freq_encoding...') as t:
            for i in t:
                try:
                    col = columns[i]
                    freqs_cat = train_data.groupby(col)[col].count() / nsamples
                    encoded_train = train_data[col].map(freqs_cat)
                    encoded_test = test_data[col].map(freqs_cat)
                    encoded_col = pd.concat([encoded_train, encoded_test], axis=0)
                    if fill_na:
                        encoded_col[encoded_col.isnull()] = 0
                    encoded_cols.append(pd.DataFrame({'freq_'+col: encoded_col}))
                except StopIteration:
                    break
    all_encoded = pd.concat(encoded_cols, axis=1)
    return all_encoded.iloc[:train_data.shape[0], :], all_encoded.iloc[train_data.shape[0]:, :]


def mean_encode(train_data, test_data, columns, target_col, reg_method=None,
                alpha=5, add_random=False, rmean=0, rstd=0.1, folds=5, seed=2019):
    """
    :return: Returns the train and test DataFrame with encoded columns
    """
    encoded_cols = []
    target_mean_global = train_data[target_col].mean()
    with tqdm(range(len(columns))) as t:
        for i in t:
            try:
                col = columns[i]
                # Getting means for test data
                nrows_cat = train_data.groupby(col)[target_col].count()
                target_means_cats = train_data.groupby(col)[target_col].mean()
                target_means_cats_adj = (target_means_cats * nrows_cat +
                                         target_mean_global * alpha) / (nrows_cat + alpha)
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
                    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
                    parts = []
                    for tr_in, val_ind in enumerate(kfold.split(train_data.values, train_data.values)):
                        # divide data
                        df_for_estimation, df_estimated = train_data.iloc[tr_in], train_data.iloc[val_ind]
                        # getting means on data for estimation (all folds except estimated)
                        nrows_cat = df_for_estimation.groupby(col)[target_col].count()
                        target_means_cats = df_for_estimation.groupby(col)[target_col].mean()
                        target_means_cats_adj = (target_means_cats * nrows_cat +
                                                 target_mean_global * alpha) / (nrows_cat + alpha)
                        # Mapping means to estimated fold
                        encoded_train_part = df_estimated[col].map(target_means_cats_adj)
                        if add_random:
                            encoded_train_part = encoded_train_part + np.random.normal(loc=rmean, scale=rstd,
                                                                                       size=(
                                                                                       encoded_train_part.shape[0]))
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
                encoded_cols.append(pd.DataFrame({'mean_' + target_col + '_' + col: encoded_col}))
            except StopIteration:
                break
    all_encoded = pd.concat(encoded_cols, axis=1)
    return all_encoded.iloc[:train_data.shape[0], :], all_encoded.iloc[train_data.shape[0]:, :]


def bayesian_target_encoding(train_data, valid_data, test_data, columns, target_col, N_min,
                             stat_type='mean', encode_na=False):
    """
    :param target_col: label
    :param N_min: regularization term, N_min 越大，regularization效果越强。
    :param stat_type: 包括 "mean" "mode" "median" "var" "skewness" "kurtosis"
    :param encode_na: 是否对nan进行编码
    :return:Returns the train and test DataFrame with encoded columns
    """
    if not encode_na:
        for col in columns:
            if train_data[col].isna().sum() and valid_data[col].isna().sum() and test_data[col].isna().sum():
                raise ValueError("the col {} contains nan".format({col}))
    else:
        train_data[columns] = train_data[columns].astype(str)
        valid_data[columns] = valid_data[columns].astype(str)
        test_data[columns] = test_data[columns].astype(str)
    feature_cols = []
    prior_mean = np.mean(train_data[target_col])
    with tqdm(range(len(columns))) as t:
        for i in t:
            try:
                col = columns[i]
                new_col = col + '_' + stat_type
                feature_cols.append(new_col)
                stats = train_data[[col, target_col]].groupby(col).agg(['sum', 'count'])[target_col].reset_index()
                train_data[new_col] = _bayesian_target_encoding_trans(train_data[[col]], stat_type,
                                                                                stats, prior_mean, N_min)
                valid_data[new_col] = _bayesian_target_encoding_trans(valid_data[[col]], stat_type,
                                                                                stats, prior_mean, N_min)
                test_data[new_col] = _bayesian_target_encoding_trans(test_data[[col]], stat_type,
                                                                                stats, prior_mean, N_min)
            except StopIteration:
                break
    if encode_na:
        train_data[columns] = train_data[columns].astype('category')
        valid_data[columns] = valid_data[columns].astype('category')
        test_data[columns] = test_data[columns].astype('category')
    return train_data, valid_data, test_data, feature_cols


def _bayesian_target_encoding_trans(data, stat_type, stats, prior_mean, N_min):
    df_stats = pd.merge(data, stats, how='left')
    df_stats['sum'].fillna(value=prior_mean, inplace=True)
    df_stats['count'].fillna(value=1.0, inplace=True)
    N_prior = np.maximum(N_min - df_stats['count'].values, 0)
    alpha_prior = prior_mean * N_prior
    beta_prior = (1 - prior_mean) * N_prior

    alpha = alpha_prior + df_stats['sum']
    beta = beta_prior + df_stats['count'] - df_stats['sum']

    if stat_type == 'mean':
        num = alpha
        dem = alpha + beta
    elif stat_type == 'mode':
        num = alpha - 1
        dem = alpha + beta - 2
    elif stat_type == 'median':
        num = alpha - 1 / 3
        dem = alpha + beta - 2 / 3
    elif stat_type == 'var':
        num = alpha * beta
        dem = (alpha + beta) ** 2 * (alpha + beta + 1)
    elif stat_type == 'skewness':
        num = 2 * (beta - alpha) * np.sqrt(alpha + beta + 1)
        dem = (alpha + beta + 2) * np.sqrt(alpha * beta)
    elif stat_type == 'kurtosis':
        num = 6 * (alpha - beta) ** 2 * (alpha + beta + 1) - alpha * beta * (alpha + beta + 2)
        dem = alpha * beta * (alpha + beta + 2) * (alpha + beta + 3)
    else:
        num = prior_mean
        dem = np.ones_like(N_prior)
    value = num / dem
    value[np.isnan(value)] = np.nanmedian(value)
    return value


def nan_encoding(train_data, test_data, columns):
    """
    :return: Returns the train and test DataFrame with encoded columns
    """
    all_data = pd.concat([train_data, test_data], axis=0)
    encoded_cols = []
    with tqdm(range(len(columns)), 'nan_encoding...') as t:
        for i in t:
            try:
                col = columns[i]
                encoded_cols.append(pd.get_dummies(all_data[col], prefix='nan_' + col, drop_first=True,
                                                   dummy_na=True))
            except StopIteration:
                break
    all_encoded = pd.concat(encoded_cols, axis=1)
    return all_encoded.iloc[:train_data.shape[0], :], all_encoded.iloc[train_data.shape[0]:, :]


def count_encoding(train_data, test_data, columns, is_concat=False, fill_na=True):
    """
    :param: fill_na: if fill_na == True will fill nan with 0, default True
    :param: is_concat: encoding时是否合并train_data和test_data
    :return: Returns the train and test DataFrame with encoded columns
    """
    encoded_cols = []
    if is_concat:
        all_data = pd.concat([train_data, test_data], axis=0)
        with tqdm(range(len(columns)), 'count_encoding...') as t:
            for i in t:
                try:
                    col = columns[i]
                    count_cat = all_data.groupby(col)[col].count()
                    encoded_col = all_data[col].map(count_cat)
                    if fill_na:
                        encoded_col[encoded_col.isnull()] = 0
                    encoded_cols.append(pd.DataFrame({'count_' + col: encoded_col}))
                except StopIteration:
                    break
    else:
        with tqdm(range(len(columns)), 'freq_encoding...') as t:
            for i in t:
                try:
                    col = columns[i]
                    count_cat = train_data.groupby(col)[col].count()
                    encoded_train = train_data[col].map(count_cat)
                    encoded_test = test_data[col].map(count_cat)
                    encoded_col = pd.concat([encoded_train, encoded_test], axis=0)
                    if fill_na:
                        encoded_col[encoded_col.isnull()] = 0
                    encoded_cols.append(pd.DataFrame({'count_' + col: encoded_col}))
                except StopIteration:
                    break
    all_encoded = pd.concat(encoded_cols, axis=1)
    return all_encoded.iloc[:train_data.shape[0], :], all_encoded.iloc[train_data.shape[0]:, :]


class BetaEncoder(object):

    def __init__(self, group, encoding_nan=False):

        self.group = group
        self.stats = None
        self.encoding_nan = encoding_nan

    # get counts from df
    def fit(self, df, target_col):
        self.prior_mean = np.mean(df[target_col])
        tmp = df[[target_col, self.group]].copy()
        if self.encoding_nan:
            tmp[self.group] = tmp[self.group].astype(str)
        stats = tmp[[target_col, self.group]].groupby(self.group)
        stats = stats.agg(['sum', 'count'])[target_col]
        stats.rename(columns={'sum': 'n', 'count': 'N'}, inplace=True)
        stats.reset_index(level=0, inplace=True)
        self.stats = stats

    # extract posterior statistics
    def transform(self, df, stat_type, N_min=1):
        tmp = df[[self.group]].copy()
        if self.encoding_nan:
            tmp[self.group] = tmp[self.group].astype(str)
        df_stats = pd.merge(tmp[[self.group]], self.stats, how='left')
        n = df_stats['n'].copy()
        N = df_stats['N'].copy()

        # fill in missing
        nan_indexs = np.isnan(n)
        n[nan_indexs] = self.prior_mean
        N[nan_indexs] = 1.0

        # prior parameters
        N_prior = np.maximum(N_min - N, 0)
        alpha_prior = self.prior_mean * N_prior
        beta_prior = (1 - self.prior_mean) * N_prior

        # posterior parameters
        alpha = alpha_prior + n
        beta = beta_prior + N - n

        # calculate statistics
        if stat_type == 'mean':
            num = alpha
            dem = alpha + beta

        elif stat_type == 'mode':
            num = alpha - 1
            dem = alpha + beta - 2

        elif stat_type == 'median':
            num = alpha - 1 / 3
            dem = alpha + beta - 2 / 3

        elif stat_type == 'var':
            num = alpha * beta
            dem = (alpha + beta) ** 2 * (alpha + beta + 1)

        elif stat_type == 'skewness':
            num = 2 * (beta - alpha) * np.sqrt(alpha + beta + 1)
            dem = (alpha + beta + 2) * np.sqrt(alpha * beta)

        elif stat_type == 'kurtosis':
            num = 6 * (alpha - beta) ** 2 * (alpha + beta + 1) - alpha * beta * (alpha + beta + 2)
            dem = alpha * beta * (alpha + beta + 2) * (alpha + beta + 3)

        else:
            num = self.prior_mean
            dem = np.ones_like(N_prior)

        # replace missing
        value = num / dem
        value[np.isnan(value)] = np.nanmedian(value)
        return value
