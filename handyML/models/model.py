"""
Authors:
        ITryagain <long452a@163.com>

Reference:
        https://www.ibm.com/developerworks/community/blogs/jfp/entry/Fast_Computation_of_AUC_ROC_score?lang=en
        https://www.kaggle.com/uberkinder/efficient-metric
        https://www.kaggle.com/artgor
introduce:
        this file contains the use of models such as LightGBM, XGBoost, CatBoost and so on
"""
import time
import gc
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn import metrics
from numba import jit
from preprocessing.Encoding import BetaEncoder
from tqdm import tqdm


# compute auc faster than sklearn
# However, when there are ties in predictions computed value may differ from sklearn value.
# https://www.ibm.com/developerworks/community/blogs/jfp/entry/Fast_Computation_of_AUC_ROC_score?lang=en
@jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


# define metric by ourself
def eval_auc(preds, dtrain):
    return 'auc', fast_auc(dtrain, preds), True


def group_mean_log_mae(y_true, y_pred, group, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true - y_pred).abs().groupby(group).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()


def train_model_regression(X, X_test, target_col, params, folds, model_type='lgb', eval_metric='mae', columns=None,
                           model=None, verbose=1000, early_stopping_rounds=200, n_estimators=50000, metrics_dict=None,
                           beta_encoding=False, cat_col=None, N_min=1, encoding_nan=False, encoding_type='mean'):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    :param: X - training data, can be pd.DataFrame (after normalizing)
    :param: X_test - test data, can be pd.DataFrame (after normalizing)
    :param: target_col - target col name
    :param: folds - folds to split data
    :param: model_type - type of model to use
    :param: eval_metric - metric to use
    :param: columns - columns to use. If None - use all columns
    :param: plot_feature_importance - whether to plot feature importance of LGB
    :param: model - sklearn model, works only for "sklearn" model type
    :param: beta_encoding - do beta_encoding in k-folds

    """
    if beta_encoding:
        if (not isinstance(cat_col, list)) and (not isinstance(cat_col, np.ndarray)):
            raise TypeError('cat_col should be list or np.ndarry')

    if columns is None:
        columns = [col for col in X.columns if col != target_col]

    if metrics_dict is None:
        metrics_dict = {
            'mae': {
                'lgb_metric_name': 'mae',
                'catboost_metric_name': 'MAE',
                'sklearn_scoring_function': metrics.mean_absolute_error
            },
            'group_mae': {
                'lgb_metric_name': 'mae',
                'catboost_metric_name': 'MAE',
                'scoring_function': group_mean_log_mae
            },
            'mse': {
                'lgb_metric_name': 'mse',
                'catboost_metric_name': 'MSE',
                'sklearn_scoring_function': metrics.mean_squared_error
            }
        }

    result_dict = {}
    # list of scores on folds
    models = []
    scores = []  # list of scores on folds
    oof = np.zeros(len(X))  # out-of-fold predictions on train data
    prediction = np.zeros(len(X_test))  # averaged predictions on train data
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X[columns], X[target_col])):
        print('Fold {} started at {}'.format({fold_n + 1}, {time.ctime()}))
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

        if beta_encoding:
            # encode variables
            feature_col = []
            for var_name in tqdm(cat_col):
                # fit encoder
                be = BetaEncoder(var_name, encoding_nan)
                be.fit(X_train[[var_name, target_col]], target_col)

                feature_name = var_name + encoding_type
                X_train[feature_name] = be.transform(X_train[[var_name]], encoding_type, N_min)
                X_valid[feature_name] = be.transform(X_valid[[var_name]], encoding_type, N_min)
                X_test[feature_name] = be.transform(X_test[[var_name]], encoding_type, N_min)
                feature_col.append(feature_name)
                gc.collect()
            columns += feature_col

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators=n_estimators, n_jobs=-1)
            model.fit(X_train[columns], X_train[target_col],
                      eval_set=[(X_train[columns], X_train[target_col]), (X_valid[columns], X_valid[target_col])],
                      eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                      verbose=verbose,
                      early_stopping_rounds=early_stopping_rounds)

            y_pred_valid = model.predict(X_valid[columns])
            y_pred = model.predict(X_test[columns], num_iteration=model.best_iteration_)
        elif model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train[columns], label=X_train[target_col], feature_names=columns)
            valid_data = xgb.DMatrix(data=X_valid[columns], label=X_valid[target_col], feature_names=columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid')]
            model = xgb.train(dtrain=train_data,
                              num_boost_round=n_estimators,
                              evals=watchlist,
                              early_stopping_rounds=early_stopping_rounds,
                              verbose_eval=verbose,
                              params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns),
                                         ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        elif model_type == 'cat':
            model = CatBoostRegressor(iterations=n_estimators,
                                      eval_metric=metrics_dict[eval_metric]['catboost_metric_name'],
                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'],
                                      verbose=verbose,
                                      cat_features=cat_col,
                                      **params)
            model.fit(X_train[columns], X_train[target_col], eval_set=(X_valid[columns], X_valid[target_col]))
            gc.collect()

            y_pred_valid = model.predict(X_valid[columns])
            y_pred = model.predict(X_test[columns])
        elif model_type == 'sklearn':
            model = model
            model.fit(X_train[columns], X_train[target_col])

            y_pred_valid = model.predict(X_valid[columns])
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](X_valid[target_col], y_pred_valid)
            print("Fold {}. {}: {}.".format({fold_n}, {eval_metric}, {score}))
            print('')

            y_pred = model.predict(X_test[columns])

        oof[valid_index] = y_pred_valid.reshape(-1,)
        if eval_metric != 'group_mae':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](X_valid[target_col], y_pred_valid))
        else:
            scores.append(metrics_dict[eval_metric]['scoring_function'](X_valid[target_col], y_pred_valid,
                                                                        X_valid['group']))
        prediction += y_pred
        models.append(model)
        gc.collect()

    prediction /= folds.n_splits
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores

    return result_dict, models


def train_model_classification(X, X_test, target_col, params, folds, model_type='lgb', eval_metric='auc', columns=None,
                           model=None, verbose=1000, early_stopping_rounds=200, n_estimators=50000, metrics_dict=None,
                           beta_encoding=False, cat_col=None, N_min=1, encoding_nan=False, encoding_type='mean'):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    :param: X - training data, can be pd.DataFrame (after normalizing)
    :param: X_test - test data, can be pd.DataFrame (after normalizing)
    :param: target_col - target col name
    :param: folds - folds to split data
    :param: model_type - type of model to use
    :param: eval_metric - metric to use
    :param: columns - columns to use. If None - use all columns
    :param: plot_feature_importance - whether to plot feature importance of LGB
    :param: model - sklearn model, works only for "sklearn" model type
    :param: beta_encoding - do beta_encoding in k-folds

    """
    if beta_encoding:
        if (not isinstance(cat_col, list)) and (not isinstance(cat_col, np.ndarray)):
            raise TypeError('cat_col should be list or np.ndarry')

    if columns is None:
        columns = [col for col in X.columns if col != target_col]

    if metrics_dict is None:
        metrics_dict = {
            'auc': {
                'lgb_metric_name': eval_auc,
                'catboost_metric_name': 'AUC',
                'sklearn_scoring_function': metrics.roc_auc_score
            },
        }

    result_dict = {}
    # list of scores on folds
    scores = []  # list of scores on folds
    models = []
    oof = np.zeros(len(X))  # out-of-fold predictions on train data
    prediction = np.zeros(len(X_test))  # averaged predictions on train data
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X[columns], X[target_col])):
        print('Fold {} started at {}'.format({fold_n + 1}, {time.ctime()}))
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

        if beta_encoding:
            # encode variables
            feature_col = []
            for var_name in tqdm(cat_col):
                # fit encoder
                be = BetaEncoder(var_name, encoding_nan)
                be.fit(X_train[[var_name, target_col]], target_col)

                feature_name = var_name + encoding_type
                X_train[feature_name] = be.transform(X_train[[var_name]], encoding_type, N_min)
                X_valid[feature_name] = be.transform(X_valid[[var_name]], encoding_type, N_min)
                X_test[feature_name] = be.transform(X_test[[var_name]], encoding_type, N_min)
                feature_col.append(feature_name)
                gc.collect()
            columns += feature_col

        if model_type == 'lgb':
            model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs=-1)
            model.fit(X_train[columns], X_train[target_col],
                      eval_set=[(X_train[columns], X_train[target_col]), (X_valid[columns], X_valid[target_col])],
                      eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                      verbose=verbose,
                      early_stopping_rounds=early_stopping_rounds)

            y_pred_valid = model.predict_proba(X_valid[columns])
            y_pred = model.predict_proba(X_test[columns], num_iteration=model.best_iteration_)
        elif model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train[columns], label=X_train[target_col], feature_names=columns)
            valid_data = xgb.DMatrix(data=X_valid[columns], label=X_valid[target_col], feature_names=columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid')]
            model = xgb.train(dtrain=train_data,
                              num_boost_round=n_estimators,
                              evals=watchlist,
                              early_stopping_rounds=early_stopping_rounds,
                              verbose_eval=verbose,
                              params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns),
                                         ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        elif model_type == 'cat':
            model = CatBoostClassifier(iterations=n_estimators,
                                       eval_metric=metrics_dict[eval_metric]['catboost_metric_name'],
                                       loss_function=metrics_dict[eval_metric]['catboost_metric_name'],
                                       verbose=verbose,
                                       cat_features=cat_col,
                                       **params)
            model.fit(X_train[columns], X_train[target_col], eval_set=(X_valid[columns], X_valid[target_col]))
            gc.collect()

            y_pred_valid = model.predict(X_valid[columns])
            y_pred = model.predict(X_test[columns])
        elif model_type == 'sklearn':
            model = model
            model.fit(X_train[columns], X_train[target_col])

            y_pred_valid = model.predict(X_valid[columns])
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](X_valid[target_col], y_pred_valid)
            print("Fold {}. {}: {}.".format({fold_n}, {eval_metric}, {score}))
            print('')

            y_pred = model.predict_proba(X_test[columns])

        oof[valid_index] = y_pred_valid.reshape(-1,)
        if eval_metric != 'group_mae':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](X_valid[target_col], y_pred_valid))
        else:
            scores.append(metrics_dict[eval_metric]['scoring_function'](X_valid[target_col], y_pred_valid,
                                                                        X_valid['group']))
        prediction += y_pred
        models.append(model)
        gc.collect()

    prediction /= folds.n_splits
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores

    return result_dict, models
