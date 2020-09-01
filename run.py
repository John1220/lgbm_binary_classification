# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import logit

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
features = [x for x in train_df.columns if x.startswith("var")]     # 特征名var_0, var_1, ...

hist_df = pd.DataFrame()
for var in features:
    var_stats = train_df[var].append(test_df[var]).value_counts()   # 每一特征取值出现的频率
    hist_df[var] = pd.Series(test_df[var]).map(var_stats)           # 只用test_df，因为train_df存在fake data
    hist_df[var] = hist_df[var] > 1                                 # 针对fake data的操作

    ind = hist_df.sum(axis=1) != 200
    # var_stats : {var_0: {num: count, num_1: count_1, ...}, ...}
    var_stats = {var: train_df[var].append(test_df[ind][var]).value_counts() for var in features}
    pred = 0

    for var in features:
        model = lgb.LGBMClassifier(**{
            'learning_rate': 0.05, 'max_bin': 165, 'max_depth': 5, 'min_child_samples': 150,
            'min_child_weight': 0.1, 'min_split_gain': 0.0018, 'n_estimators': 41,
            'num_leaves': 6, 'reg_alpha': 2.0, 'reg_lambda': 2.54, 'objective': 'binary', 'n_jobs': -1})
        # X : 每条样本为 [var_0_num, var_0_count]
        X = [train_df[var].values.reshape(-1, 1), train_df[var].map(var_stats[var]).values.reshape(-1, 1)]
        model = model.fit(np.hstack(X), train_df["target"].values)
        # 对每个特征，将预测概率加到pred汇总
        # logit: make (0,1) -> (-inf, inf)
        pred += logit(model.predict_proba(np.hstack([test_df[var].values.reshape(-1, 1),
                                                     test_df[var].map(var_stats[var]).values.reshape(-1, 1)]))[:, 1])
    # 评价基于AUC，只需要输出样本之间概率的相对大小
    pd.DataFrame({"ID_code": test_df["ID_code"], "target": pred}).to_csv("result.csv", index=False)