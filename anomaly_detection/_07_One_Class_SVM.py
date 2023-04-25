# 원-클래스 SVM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

rng = np.random.RandomState(50)
# 훈련데이터 생성
X = 0.3*rng.randn(100, 2)
X_train = np.r_[X+2, X-2]
X_train = pd.DataFrame(X_train, columns = ['x1','x2'])

X = 0.3*rng.randn(20, 2)
X_test = np.r_[X+2, X-2]
X_test = pd.DataFrame(X_test, columns=['x1','x2'])

X_outliers = rng.uniform(low=-4, high=4, size=(20,2))
X_outliers = pd.DataFrame(X_outliers, columns = ['x1','x2'])

import matplotlib.pyplot as plt
plt.style.use(['dark_background'])

plt.rcParams['figure.figsize'] = [10,10]

# 정상 훈련 데이터 (하양)
p1=plt.scatter(X_train.x1, X_train.x2, c='white', s=20*2, edgecolor='k', label='training observations')

# 정상샘플에 가까운 테스트 셋 (초록)
p2=plt.scatter(X_test.x1, X_test.x2, c='green', s=20*2, edgecolor='k', label='new regular obs.')

# 아웃라이어 샘플 (빨강)
p3=plt.scatter(X_outliers.x1, X_outliers.x2, c='red', s=20*2, edgecolor='k', label='new abnormal obs.')

plt.legend()
# plt.gcf().set_size_inches(5,5)
# plt.show()

# 모델 학습 및 평가
# Kernel : 하이퍼 플랜의 종류
# gamma : 서포트 벡터와의 거리, 클수록 가까이 있는 데이터
# nu : 하이퍼 플랜 밖에 있는 데이터의 비율(abnormal)

clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)        # 학습
y_pred_train = clf.predict(X_train)     # 예측
y_pred_test = clf.predict(X_test)       # 예측
y_pred_outliers = clf.predict(X_outliers)       # 예측

# 예측데이터 시각화
X_outliers = X_outliers.assign(y=y_pred_outliers)

p1 = plt.scatter(X_train.x1, X_train.x2, c='white', s= 20*2,
edgecolor = 'k', label="training observations")

p2 = plt.scatter(X_outliers.loc[X_outliers.y == -1, ['x1']],
X_outliers.loc[X_outliers.y == -1, ['x2']],
c='red',s=20*2, edgecolor='k', label="detected outliers")

p3 = plt.scatter(X_outliers.loc[X_outliers.y == 1, ['x1']],
X_outliers.loc[X_outliers.y == 1, ['x2']],
c='green', s=20*2, edgecolor='k', label="detected regular obs")

plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5));
plt.gcf().set_size_inches(10,10)
# plt.show()

# 정상 데이터 셋을 얼만큼 정상으로 예측하였는가?
print("테스트 데이터셋에서 정확도:", list(y_pred_test).count(1) / y_pred_test.shape[0])

# Outlier 데이터 셋을 얼마나 Outlier로 예측하였는가?
print("이상치 데이터셋에서 정확도:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])

# gamma 값을 수정해가면서 제일 잘 구분하는 하이퍼 파라미터를 찾아야한다
