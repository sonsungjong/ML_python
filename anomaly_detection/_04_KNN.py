# KNN 이상 탐지 (거리와 밀도기반 이상탐지 방법론)

# KNN : K-Nearest_Neighbors, K개의 이웃한 데이터를 기반으로 대상 분류
# 지도학습 기반
# 주로 EDA 또는 Base-Line을 잡기 위해 사용하는 알고리즘
# 탑재로써는 성능이 떨어짐

'''
KNN 장단점
+ 높은 정확도
+ 수치기반 데이터 분류 작업에서 우수한 성능 (매출, 고객수 등 수치기반 분류)
+ 데이터에 대한 정규성이나 선형조건이 없어도 사용 가능
+ 이미지처리, 글자/얼굴 인식, 상품추천, 패턴인식, 이상탐지

- 데이터가 많을수록 처리 속도가 느림
- 거리가 가까운 이유 및 상관관계를 알 수 없음 (분포기반)
- 카테고리컬 데이터를 위한 추가 처리가 필요함
'''

'''
KNN 사용법

- PyOD 패키지 활용 (pip install pyod), 이상탐지 알고리즘 패키지
- K 갯수 결정
- 표준화
- 새로운 Data로 지속 수행
'''

from pyod.utils.example import visualize
from pyod.utils.data import evaluate_print
from pyod.utils.data import generate_data
from pyod.models.knn import KNN

contamination = 0.1             # percentage of outliers
n_train = 200                   # number of training points
n_test = 100                    # number of testing points

# Generate sample data
X_train, X_test, y_train, y_test = generate_data(n_train=n_train, n_test=n_test, n_features=2, contamination=contamination, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# 모델 생성 및 학습
clf_name = 'KNN'
clf = KNN()
clf.fit(X_train)

# Train set score
y_train_pred = clf.labels_
y_train_scores = clf.decision_scores_

# Test set score
y_test_pred = clf.predict(X_test)
y_test_scores = clf.decision_function(X_test)

print('On Training Data:')
evaluate_print(clf_name, y_train, y_train_scores)

print('\nOn Test Data:')
evaluate_print(clf_name, y_test, y_test_scores)

# visualize the results
visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, show_figure=True, save_figure=True)
