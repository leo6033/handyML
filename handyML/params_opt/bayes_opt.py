"""
Authors:
        Zwp <570972004@qq.com>

Referance:
    https://github.com/fmfn/BayesianOptimization

introduce:
        使用贝叶斯优化调整参数，是对bayes_opt的高层封装

"""
from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd

class ParamOptimizer(object):
    def __init__(self, score_func, score_func_parmas):
        """
        :param: score_func: 计算得分的函数 handyML.models.model.train_model_regression / handyML.models.model.train_model_regression
        :param: score_func_parmas: 模型函数的参数
        """
        self.score_func = score_func
        self.score_func_parmas = score_func_parmas
        
        self.solid_params = {} #恒定的训练参数
        self.optimizer = None
        self.int_params = None
    
    def optimize(self, params, int_params=None):
        """
        :param: params: 参数字典  {'num_leaves':1, 'max_depth':(1,2)}
        :param: int_params: 参数字典中整型参数列表
        """
        self.int_params = int_params
        opt_params = {}
        for k in params.keys():
            if isinstance(params[k], tuple) or isinstance(params[k], list):
                opt_params[k] = params[k]
            else:
                self.solid_params[k] = params[k]
        self.optimizer = BayesianOptimization(self._optimize_func, opt_params, random_state=11)
        self.optimizer.maximize()
        
        print("Best score:",self.optimizer.max['target'])
        return self.optimizer.max['params']
    
    def _optimize_func(self,**params):
        """
        内部优化函数，返回当前参数下的local CV
        """
        p = self.solid_params.copy()
        for k in params.keys():
            if k in self.int_params:
                p[k] = int(round(params[k]))
            else:
                p[k] = params[k]
        print(p['min_data_in_leaf'])
        self.score_func_parmas['params'] = p
        result_dict, _ = self.score_func(**self.score_func_parmas)
        return np.mean(result_dict['scores'])



