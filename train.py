import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
train_meta_df = pd.read_csv("greeks.csv")
# 缺失值处理
imputer = SimpleImputer(strategy='median')
train_df['EJ'] = train_df['EJ'].fillna(train_df['EJ'].mode()[0])
for col in train_df.columns:
    if col not in ['Id', 'Class', 'EJ'] and train_df[col].isnull().any():
        imputer.fit(train_df[[col]])
        train_df[col] = imputer.transform(train_df[[col]]).ravel()
# 编码
le = LabelEncoder()
train_df['EJ'] = le.fit_transform(train_df['EJ'])
train_df['Class'] = train_df['Class'].astype(int)
train_meta_df['Alpha'] = le.fit_transform(train_meta_df['Alpha'].astype(str))

# 去除id和class列
X_train = train_df.drop(columns=['Id', 'Class'])
y_train = train_df['Class']

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# 训练模型

clf = XGBClassifier()
clf.fit(X_train, y_train)

# 交叉验证评估模型
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# 预测test数据集
test_df['EJ'] = test_df['EJ'].fillna(test_df['EJ'].mode()[0])
for col in test_df.columns:
    if col not in ['Id', 'EJ'] and test_df[col].isnull().any():
        imputer.fit(test_df[[col]])
        test_df[col] = imputer.transform(test_df[[col]]).ravel()
test_df['EJ'] = le.transform(test_df['EJ'])
X_test = test_df.drop(columns=['Id'])
y_pred = clf.predict_proba(X_test)

# 制作提交文件
submission_df = pd.DataFrame({'Id': test_df.Id, 'Class_0': y_pred[:,0], 'Class_1': y_pred[:,1]})
submission_df.to_csv('submission.csv', index=False)