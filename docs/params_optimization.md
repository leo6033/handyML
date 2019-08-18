# Parameters Optimization

------

### class handyML.params_opt.bayes_opt.**ParamOptimizer**(*score_func, score_func_params*)

该类使用贝叶斯优化调整模型参数。

Referance:

​	[GitHub: Bayesian Optimization](https://github.com/fmfn/BayesianOptimization)

------

#### **\_\_init\_\_** *(self, score_func, score_func_parmas)*

**参数：**

- **score_func : *function***

  模型函数，以下二者之一：

  - handyML.models.model.train_model_regression
  - handyML.models.model.train_model_regression

- **score_func_parmas : *dict***

  模型函数的参数字典

#### **optimize** *(self, params, int_params=None)*

优化函数

**参数：**

- **params :  *dict***

  参数字典，包含了要调参的参数和已经设置好的参数。

  要调参的参数使用`tuple`或`list`来限制参数取值。

- **int_params : *list, default=None***

  参数字典中类型为整型的待优化参数列表

  

```python
from handyML.params_opt.bayes_opt import ParamOptimizer
from handyML.models.model import train_model_classification
from handyML.tests import load_data
from sklearn.model_selection import KFold

train = load_data("train_classification")
test = load_data("test_classification")

folds = KFold(n_splits=5, shuffle=True, random_state=2019)
score_func_parmas = {
    'X':train,
    'X_test':test,
    'target_col':'favorite',
    'params':None,
    'folds':folds,
    'columns':feats,
}

param = {
    'boost': 'gbdt',
    'feature_fraction': 0.8,
    'feature_fraction_seed':11,
    'learning_rate': 0.01,
    'metric':'auc',
    'num_leaves': 25,
    'objective': 'binary', 
    "lambda_l1": 0.3,
    #---------------------------------------
    'min_data_in_leaf': (1, 20),
    'max_depth': (3, 9),  
}

po = ParamOptimizer(train_model_classification, score_func_parmas)
po.optimize(param,['min_data_in_leaf','max_depth'])
```

