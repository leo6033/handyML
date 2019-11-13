# models

## regression

```
train_model_regression(X, X_test, target_col, params, folds, model_type='lgb', eval_metric='mae', columns=None,
                           model=None, verbose=1000, early_stopping_rounds=200, n_estimators=50000, metrics_dict=None,
                           beta_encoding=False, cat_col=None, encode_col=None, N_min=1, encoding_nan=False, encoding_type='mean',
                           feature_importance=True)
```

该函数用于回归问题，封装模型包括（LightGBM、CatBoost、XGBoost 和 sklearn中的模型）

参数介绍如下：

+ **X ：训练集**
+ **X_test ：测试集**
+ **target_col ：label所在列列名**
+ **params ： 模型所需参数**
+ **folds ：交叉验证，输入可以为sklearn中的 KFold、StratifiedKFold等对象**
+ **model_type ：模型类型 （'lgb', 'xgb', 'cat' 和 'sklearn'）**
+ **eval_metric ： 评估指标**
+ **columns ： 选用的特征列，若不输入，则默认为使用训练集中所有列**
+ **model ：默认为None，当选用model_type为 sklearn 时，需输入sklearn模型**
+ **verbose、 early_stopping_rounds、n_estimators、 cat_col ： 与LightGBM、CatBoost、XGBoost中参数相同**
+ **metrics_dict ： 评估字典，当需要使用自定以评估指标时使用**
+ **beta_encoding ： 是否使用 beta_encoding**
+ **encode_col ： 需要做encoding的列**
+ **N_min ：beta_encoding中的一个参数，是一个regularization term，N_min 越大，regularization效果越强**
+ **encoding_type ： beta_encoding 的类型，包括有 mean、mode、median、var、skewness、kurtosis**
+ **feature_importance ：是否输出特征重要性，模型为LightGBM、CatBoost、XGBoost的时候使用**

返回：

+ result_dict ：一个包含有 oof、predict和模型得分的字典
+ feature_important_df ：包含有特征重要性的DataFrame

## classification

```
train_model_classification(X, X_test, target_col, params, folds, model_type='lgb', eval_metric='auc', columns=None,
                           model=None, verbose=1000, early_stopping_rounds=200, n_estimators=50000, metrics_dict=None,
                           beta_encoding=False, cat_col=None, encode_col=None, N_min=1, encoding_nan=False, encoding_type='mean',
                           feature_importance=False)
```

该函数用于回归问题，封装模型包括（LightGBM、CatBoost、XGBoost 和 sklearn中的模型），参数介绍与回归函数相同。

## 用法

（以 kaggle IEEE-CIS Fraud Detection 为例）

```python
params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'n_jobs':-1,
                    'learning_rate':0.01,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':1,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': 42,
                } 

encod_col = ['ProductCD','card5',
             'addr1', 
             'P_emaildomain', 'R_emaildomain', 
             'DeviceInfo']

folds = KFold(n_splits=5)
X = pd.concat([X, y], axis=1)
result_dict, feature_importance_df = train_model_classification(X, X_test,target_col, params, folds, columns=features_columns, verbose=500, beta_encoding=True, cat_col=encod_col, feature_importance=True)
```

输出：

```
Fold {1} started at {'Thu Aug 22 17:37:27 2019'}
```



```
100%|████████████████████████████████████████████████████████████████████████████████████| 6/6 [05:00<00:00, 50.05s/it]
```



```
Training until validation scores don't improve for 200 rounds.
[500]	valid_0's auc: 0.995958	valid_0's auc: 0.995958	valid_1's auc: 0.921403	valid_1's auc: 0.921403
Early stopping, best iteration is:
[492]	valid_0's auc: 0.995747	valid_0's auc: 0.995747	valid_1's auc: 0.921583	valid_1's auc: 0.921583
Fold {2} started at {'Thu Aug 22 17:52:39 2019'}
```



```
100%|████████████████████████████████████████████████████████████████████████████████████| 6/6 [05:01<00:00, 50.33s/it]
```



```
Training until validation scores don't improve for 200 rounds.
[500]	valid_0's auc: 0.996645	valid_0's auc: 0.996645	valid_1's auc: 0.92816	valid_1's auc: 0.92816
Early stopping, best iteration is:
[553]	valid_0's auc: 0.997687	valid_0's auc: 0.997687	valid_1's auc: 0.928496	valid_1's auc: 0.928496
Fold {3} started at {'Thu Aug 22 18:08:52 2019'}
```



```
100%|████████████████████████████████████████████████████████████████████████████████████| 6/6 [05:00<00:00, 50.10s/it]
```



```
Training until validation scores don't improve for 200 rounds.
[500]	valid_0's auc: 0.996861	valid_0's auc: 0.996861	valid_1's auc: 0.930303	valid_1's auc: 0.930303
Early stopping, best iteration is:
[451]	valid_0's auc: 0.995944	valid_0's auc: 0.995944	valid_1's auc: 0.93054	valid_1's auc: 0.93054
Fold {4} started at {'Thu Aug 22 18:23:41 2019'}
```



```
100%|████████████████████████████████████████████████████████████████████████████████████| 6/6 [04:56<00:00, 49.42s/it]
```



```
Training until validation scores don't improve for 200 rounds.
[500]	valid_0's auc: 0.996646	valid_0's auc: 0.996646	valid_1's auc: 0.945427	valid_1's auc: 0.945427
Early stopping, best iteration is:
[544]	valid_0's auc: 0.997563	valid_0's auc: 0.997563	valid_1's auc: 0.94591	valid_1's auc: 0.94591
Fold {5} started at {'Thu Aug 22 18:39:52 2019'}
```



```
100%|████████████████████████████████████████████████████████████████████████████████████| 6/6 [04:53<00:00, 49.11s/it]
```



```
Training until validation scores don't improve for 200 rounds.
[500]	valid_0's auc: 0.996714	valid_0's auc: 0.996714	valid_1's auc: 0.927761	valid_1's auc: 0.927761
Early stopping, best iteration is:
[417]	valid_0's auc: 0.994886	valid_0's auc: 0.994886	valid_1's auc: 0.928188	valid_1's auc: 0.928188
CV mean score: 0.9309, std: 0.0081.
```

## 自定义metric_dict格式

```
metrics_dict = {
            'auc': {
                'lgb_metric_name': eval_auc,
                'catboost_metric_name': 'AUC',
                'xgb_metric_name': 'auc',
                'sklearn_scoring_function': metrics.roc_auc_score
            },
        }
```

上面的`metrics_dict`为函数内默认的dict，当使用LightGBM，需要自定义metric时，可按如下格式

```
metrics_dict = {
            eval_metric : {
                'lgb_metric_name': metric_fun(),
            },
        }
```

