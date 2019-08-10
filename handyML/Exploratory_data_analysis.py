import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

class Overview(object):
    @staticmethod
    def check_missing_data(data):
        """
        检查数据中缺失值
        
        :param: data:输入数据
        :resturn :缺失值信息表格
        """
        total = data.isnull().sum()
        percent = (data.isnull().sum()/data.isnull().count()*100)
        res = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        types = []
        for col in data.columns:
            dtype = str(data[col].dtype)
            types.append(dtype)
        res['Types'] = types
        return res
    
    @staticmethod
    def plot_feature_distribution(df, y_features, x_feature=None, hue=None, columns=2):
        """
        画m×n的折线图/密度图
        :param: df:数据
        :param: y_features:纵坐标特征列表（list，长度小于等于100）
        :param: x_feature:横坐标的特征（str）
        :param: hue:分类目标特征 (str)
        :param: columns:每一行绘制的图表数量
        
        　　当传入的df为一个pd.DataFrame()对象时，每一个图表仅绘制该对象的数据信息；当df为一个字典时(
        字典的键应为其对应的字典中的数据的标签，如：{'train':df_train,'test':df_test})，每个图表绘
        制字典中存储的所有的数据信息。
        　　当x_feature为空时，绘制y_features中每一个feature的密度图；否则绘制以x_feature为横轴的
        折线图。
        """
        #检查参数 check params
        if hue and (x_feature is None):
            raise Exception("Wrong Params!")
        
        n = columns
        hight = math.ceil(len(y_features)/n)
        
        width = 36
        height = width/n*2/3
            
        i = 0
        for feature in y_features:
            i = (i+1) % n
            if i==1:
                fig, ax = plt.subplots(1,n,figsize=(width,height))
                plt.subplot(1,n,1)
            elif i==0:
                plt.subplot(1,n,n)
            else:
                plt.subplot(1,n,i)
            if isinstance(df,dict):
                for df_ in df.values():
                    if x_feature:
                        for _ in df.keys():
                            if df[_] is df_: label = _ 
                        if hue:
                            sns.lineplot(x=x_feature, y=feature, hue=hue, data=df_, label=label)
                        else:
                            sns.lineplot(x=x_feature, y=feature, data=df_, label=label)
                    else:
                        sns.distplot(df_[feature])
                plt.legend()
                plt.xlabel(feature, fontsize=14)
            else:
                if x_feature:
                    if hue:
                        sns.lineplot(x=x_feature, y=feature, hue=hue, data=df)
                    else:
                        sns.lineplot(x=x_feature, y=feature, data=df)
                else:
                    sns.distplot(df[feature])
                plt.xlabel(feature, fontsize=14)
            
            plt.tick_params(axis='x', which='major', labelsize=14, pad=0)
            plt.tick_params(axis='y', which='major', labelsize=14)
            if i==0:
                plt.show()
