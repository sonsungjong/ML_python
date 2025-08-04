# pip install scikit-learn
# pip install xgboost

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV            # 최적 하이퍼파라미터 탐색용
import time
# 예측 시간: 0.104709초

df = pd.read_csv("mdi.csv", sep=";")
print(df.info())
print(df.head())

target_col = 'HUMAN'
X = df.drop(columns=[target_col])
y = df[target_col]

# 학습 및 테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 학습
model = XGBClassifier(
    n_estimators=30,            # 30 ~ 100
    max_depth=5,                # 3 ~ 10
    eval_metric='logloss',
    n_jobs=1,           # CPU 코어 1개만 사용
    tree_method='hist',
    random_state=42
    )
model.fit(X_train, y_train)

# 예측
start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()

# 평가
print("정확도:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"예측 시간: {end_time - start_time:.6f}초")

# precision : 정밀도 (예측한 것 중 정답 비율)
# recall : 재현율 (실제 정답 중 맞춘 비율)
# f1-score : 정밀도와 재현율의 조화 평균
# support : 실제 정답 데이터 개수 (샘플 개수)

# accuracy : 전체 정확도 (정답 맞춘 비율 0 ~ 1)
# macro avg : 클래스별로 평균 낸 값 (클래스 불균형에 영향 적음)
# weighted avg : support(표본수)로 가중평균 낸 값 (실제 데이터 분포 반영)