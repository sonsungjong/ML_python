# AutoEncoder : 들어갔던 X 데이터를 X데이터로 특징을 부활시킨다
# pip install pyod

import numpy as np
import pandas as pd
import seaborn as sns
from pyod.models.auto_encoder import AutoEncoder
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print

contamination = 0.1             # 10%가 이상데이터라고 가정
n_train = 20000
n_test = 2000
n_features = 300

X_train, X_test, y_train, y_test = generate_data(n_train=n_train, n_test=n_test, n_features=n_features, contamination=contamination, random_state=42)

# print(X_train[1])

# 모델 생성/학습 및 Hyper parameter 선정
clf_name = 'AutoEncoder'
clf = AutoEncoder(hidden_neurons=[300, 100, 100, 300], epochs=10, contamination=contamination)
clf.fit(X_train)            # 훈련시키기(학습)

y_train_pred = clf.labels_
y_train_scores = clf.decision_scores_

# 테스트 데이터에 대해 예측(분석)시작
y_test_pred = clf.predict(X_test)
y_test_scores = clf.decision_function(X_test)

# score 가 높을 수록 Outlier에 가까움
y_train_pred[0:5], y_train_scores[0:5]

# Outlier 예측 데이터 수
pd.Series(y_test_pred).value_counts()

def mod_z(col):
    med_col = col.median()
    med_abs_dev = (np.abs(col-med_col)).median()
    mod_z = 0.7413 * ((col-med_col) / med_abs_dev)
    return np.abs(mod_z)

pd_s = pd.Series(y_test_scores);
mod_z = mod_z(pd_s)
sns.distplot(mod_z)

# 성능 평가
print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nOn Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)
