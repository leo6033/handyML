import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from scipy import stats


class Overview(object):
    @staticmethod
    def check_missing_data(data):
        """
        检查数据中缺失值

        :param: data:输入数据
        :resturn :缺失值信息表格
        """
        total = data.isnull().sum()
        percent = (data.isnull().sum() / data.isnull().count() * 100)
        res = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        types = []
        for col in data.columns:
            dtype = str(data[col].dtype)
            types.append(dtype)
        res['Types'] = types
        return res

    @staticmethod
    def plot_feature_distribution(df, y_features, x_feature=None, hue=None, columns=5):
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
        # 检查参数 check params
        if hue and (x_feature is None):
            raise Exception("Wrong Params!")

        n = columns
        hight = math.ceil(len(y_features) / n)

        width = 36
        height = width / n * 2 / 3

        i = 0
        for feature in y_features:
            i = (i + 1) % n
            if i == 1:
                fig, ax = plt.subplots(1, n, figsize=(width, height))
                plt.subplot(1, n, 1)
            elif i == 0:
                plt.subplot(1, n, n)
            else:
                plt.subplot(1, n, i)
            if isinstance(df, dict):
                for df_ in df.values():
                    for _ in df.keys():
                        if df[_] is df_: label = _
                    df_ = df_[~df_[feature].isna()]

                    if x_feature:
                        df_ = df_[~df_[x_feature].isna()]
                        if hue:
                            sns.lineplot(x=x_feature, y=feature, hue=hue, data=df_, label=label)
                        else:
                            sns.lineplot(x=x_feature, y=feature, data=df_, label=label)
                    else:
                        sns.distplot(df_[feature], label=label)
                plt.legend()
                if x_feature:
                    plt.xlabel(x_feature, fontsize=14)
                else:
                    plt.xlabel(feature, fontsize=14)
            else:
                df = df[~df[feature].isna()]
                if x_feature:
                    df = df[~df[x_feature].isna()]
                    if hue:
                        sns.lineplot(x=x_feature, y=feature, hue=hue, data=df)
                    else:
                        sns.lineplot(x=x_feature, y=feature, data=df)
                    plt.xlabel(x_feature, fontsize=14)
                else:
                    sns.distplot(df[feature])
                    plt.xlabel(feature, fontsize=14)

            plt.tick_params(axis='x', which='major', labelsize=14, pad=0)
            plt.tick_params(axis='y', which='major', labelsize=14)
            if i == 0:
                plt.show()

    @staticmethod
    def describe_table(df):
        print("Dataset Shape: {}".format({df.shape}))
        summary = pd.DataFrame(df.dtypes, columns=['dtypes']).reset_index()
        summary['Name'] = summary['index']
        summary = summary[['Name', 'dtypes']]
        summary['Missing'] = df.isnull().sum().values
        summary['Uniques'] = df.nunique().values
        summary['First Value'] = df.loc[0].values
        summary['Second Value'] = df.loc[1].values
        summary['Third Value'] = df.loc[2].values

        for name in summary['Name'].value_counts().index:
            summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True),
                                                                                  base=2), 2)
        return summary

    @staticmethod
    def feature_quantiles(df, columns):
        print("Features Quantitles:")
        print(df[columns].quantile([0.01, .025, .1, .25, .5, .75, .975, .99]))

    @staticmethod
    def cal_outliers(df_num):

        # calculating mean and std of the array
        data_mean, data_std = np.mean(df_num), np.std(df_num)

        # setting the cut line to both higher and lower values
        # this value can change
        cut = data_mean * 3

        # calculating the higher and lower cut values
        lower, upper = data_mean - cut, data_mean + cut

        # creating an array of lower, higher and total outlier values
        outliers_lower = [x for x in df_num if x < lower]
        outliers_higher = [x for x in df_num if x > upper]
        outliers_total = outliers_lower + outliers_higher

        # array without outlier values
        outliers_removed = [x for x in df_num if lower < x < upper]

        print('Identified lowest outliers: {}'.format(len(outliers_lower)))
        print('Identified upper outliers: {}'.format(len(outliers_higher)))
        print('Total outlier observations: {}'.format(len(outliers_total)))
        print('Non-Outlier observations: {}'.format(len(outliers_removed)))
        print("Total percent of Outliers: ", round((len(outliers_total) / len(outliers_removed)) * 100, 4))

        return
