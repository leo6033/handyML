from sklearn import preprocessing

# 均匀分桶：主要用于将连续特征转化为类别离散特征
# 先使得特征均匀
def norm_data(df,features):   
    scaler = preprocessing.QuantileTransformer(random_state=0)
    scaler.fit(df[features].values.reshape(-1, 1)) 
    new_c = 'cat_'+features
    df[new_c]=scaler.transform(df[features].values.reshape(-1, 1))
    return df
#norm_split(train, test, 'fuzz_qratio')
#norm_data(data, 'fuzz_qratio')

# 再对均匀分布的数据分桶
# the continous_to_category function will create new category features 
def continous_to_category(data, features):
    data = norm_data(data, features)
    bins_list = []
    new_c = 'cat_'+features
    bins_num = 10
    i_min = data[new_c].min()-0.001
    i_max = data[new_c].max()+0.001
    for i in range(bins_num+1):
        bins_list.append(i_min+(i_max-i_min)/10*i)
    data[new_c] = pd.cut(data[new_c], bins=bins_list, labels=np.arange(bins_num))
    return data
# 用法如下：
# newdata = continous_to_category(data, 'feature_name')
