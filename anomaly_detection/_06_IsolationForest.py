# 가장 무난하게 이상탐지에 사용할 수 있음
# 이상값이 없어도 사용가능
# 직선으로 구간을 나누기 때문에 잘못된 예측값을 줄 수 있음
# Extended Isolation Forest 같이 직선이 아닌 도형형태로 아웃라이어를 구분하는 업그레이드 버전도 출시됨

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)
# 트레이닝 데이터 생성
X_train = 0.2 * rng.randn(1000, 2)
X_train = np.r_[X_train + 3, X_train]
X_train = pd.DataFrame(X_train, columns = ['x1','x2'])

X_test = 0.2 * rng.randn(200, 2)
X_test = np.r_[X_test + 3, X_test]
X_test = pd.DataFrame(X_test, columns = ['x1', 'x2'])

X_outliers = rng.uniform(low=-1, high=5, size=(50, 2))
X_outliers = pd.DataFrame(X_outliers, columns = ['x1', 'x2'])

import matplotlib.pyplot as plt
plt.style.use(['dark_background'])
plt.rcParams['figure.figsize'] = [10, 10]

# Train Set
p1 = plt.scatter(X_train.x1, X_train.x2, c='white', s=20*2, edgecolor='k', label='training observations')

# Test Set, 정상 샘플로 구성
p2 = plt.scatter(X_test.x1, X_test.x2, c='green', s=20*2, edgecolor='k', label='new regular obs.')

# Outlier Set
p3 = plt.scatter(X_outliers.x1, X_outliers.x2, c='red', s=20*2, edgecolor='k', label='new abnormal obs.')

plt.legend()
# plt.show()

# max_samples : 샘플링 데이터의 갯수
# contamination : 전체 데이터에서 이상치의 비율
# max_features : 학습 시 사용할 Feature
clf = IsolationForest(max_samples=100, contamination=0.1, random_state=42)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

print(y_pred_test)

# 낮을수록 Outlier(음수)
clf.decision_function(X_outliers)

# 높을수록 Inlier(양수)
clf.decision_function(X_test)[0:5]

X_outliers = X_outliers.assign(y=y_pred_outliers)

p1 = plt.scatter(X_train.x1, X_train.x2, c='white', s=20*2,
edgecolor='k', label="training observations")

p2 = plt.scatter(X_outliers.loc[X_outliers.y == -1, ['x1']],
X_outliers.loc[X_outliers.y == -1, ['x2']],
c='red', s=20*2, edgecolor='k', label="detected outliers")

p3 = plt.scatter(X_outliers.loc[X_outliers.y == 1, ['x1']],
X_outliers.loc[X_outliers.y == 1, ['x2']],
c='green', s=20*2, edgecolor='k', label="detected regular obs")

plt.legend()
plt.gcf().set_size_inches(10, 10)

# 정상 data set을 얼만큼 정상으로 예측하였는가?
print("테스트 데이터셋에서 정확도:", list(y_pred_test).count(1)/y_pred_test.shape[0])

print("이상치 데이터셋에서 정확도:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])