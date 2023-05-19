import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer,KNNImputer
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
imputer_median = SimpleImputer(strategy='median')
imputer_mean = SimpleImputer(strategy='mean')
imputer_knn = KNNImputer(n_neighbors=3, weights="uniform")
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
train_meta_df = pd.read_csv("greeks.csv")

skewed_features = ['AB', 'AH', 'AM', 'BP', 'CS']

for col in train_df.columns:
    if col not in ['Id', 'Class', 'EJ'] and train_df[col].isnull().any():
        if col in skewed_features:
            imputer_median.fit(train_df[[col]])
            train_df[col] = imputer_median.transform(train_df[[col]]).ravel()
        else:
            imputer_mean.fit(train_df[[col]])
            train_df[col] = imputer_mean.transform(train_df[[col]]).ravel()
        imputer_knn.fit(train_df[[col]])
        train_df[col] = imputer_knn.transform(train_df[[col]])[:, 0]
# 缺失值处理
imputer = SimpleImputer(strategy='median')
train_df['EJ'] = train_df['EJ'].fillna(train_df['EJ'].mode()[0])
for col in train_df.columns:
    if col not in ['Id', 'Class', 'EJ'] and train_df[col].isnull().any():
        imputer.fit(train_df[[col]])
        train_df[col] = imputer.transform(train_df[[col]]).ravel()
# 编码

encoder = OneHotEncoder(categories='auto')
encoder.fit(train_df[['EJ']])
ej_encoded = encoder.transform(train_df[['EJ']]).toarray()
train_df = pd.concat([train_df, pd.DataFrame(ej_encoded, columns=['EJ_{}'.format(i) for i in range(ej_encoded.shape[1])])], axis=1)
train_df.drop(columns=['EJ'], inplace=True)


# le = LabelEncoder()
# train_df['EJ'] = le.fit_transform(train_df['EJ'])
# train_df['Class'] = train_df['Class'].astype(int)
# train_meta_df['Alpha'] = le.fit_transform(train_meta_df['Alpha'].astype(str))

# 去除id和class列
X_train = train_df.drop(columns=['Id', 'Class'])
y_train = train_df['Class']

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# 训练模型
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.5],
    'n_estimators': [100, 500, 1000],
    'colsample_bytree': [0.5, 0.7, 0.9],
    'subsample': [0.5, 0.7, 0.9],
    'reg_alpha': [0.01, 0.1, 1],
    'reg_lambda': [0.01, 0.1, 1]
}
clf = XGBClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best hyperparameters: ", grid_search.best_params_)

# 交叉验证评估模型
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# 预测test数据集
test_df['EJ'] = test_df['EJ'].fillna(test_df['EJ'].mode()[0])
for col in test_df.columns:
    if col not in ['Id', 'EJ'] and test_df[col].isnull().any():
        imputer.fit(test_df[[col]])
        test_df[col] = imputer.transform(test_df[[col]]).ravel()
test_df['EJ'] = encoder.transform(test_df['EJ'])
X_test = test_df.drop(columns=['Id'])
y_pred = clf.predict_proba(X_test)

# 制作提交文件
submission_df = pd.DataFrame({'Id': test_df.Id, 'Class_0': y_pred[:,0], 'Class_1': y_pred[:,1]})
submission_df.to_csv('submission.csv', index=False)