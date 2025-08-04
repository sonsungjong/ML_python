# pip install scikit-learn
# pip install catboost

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV            # 최적 하이퍼파라미터 탐색용
import time
# 예측 시간: 0.033533초

df = pd.read_csv("mdi.csv", sep=";")
print(df.info())
print(df.head())

target_col = 'HUMAN'
X = df.drop(columns=[target_col])
y = df[target_col]

# 학습 및 테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CatBoost 분류기 학습
model = CatBoostClassifier(
    iterations=30,   # 트리 개수 (n_estimators)
    depth=5,         # 트리 깊이 (max_depth)
    verbose=0,       # 로그 숨김
    random_seed=42
)
model.fit(X_train, y_train)

# 예측 시간 측정
start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()

# 평가
print("정확도:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"예측 시간: {end_time - start_time:.6f}초")