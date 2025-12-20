# pip install scikit-learn
# pip install xgboost

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV            # 최적 하이퍼파라미터 탐색용
import time

df = pd.read_csv("mdi.csv", sep=",")
print(df.info())
print(df.head())

target_col = 'HUMAN'
X = df.drop(columns=[target_col])
y = df[target_col]

# 학습 및 테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

USE_GRID_SEARCH = False  # True로 변경하면 GridSearchCV 탐색 실행
# 하이퍼파라미터 초기값
g_n_estimators = 50
g_max_depth = 7
g_learning_rate = 0.1


if USE_GRID_SEARCH:
    print("="*60)
    print("하이퍼파라미터 탐색 시작 (시간 오래 걸림)...")
    print("="*60)
    
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [7, 10, 12, 15],
        'learning_rate': [0.1, 0.05, 0.03]
    }
    
    base_model = XGBClassifier(tree_method='hist', random_state=42)
    
    grid_search = GridSearchCV(
        estimator=base_model,       # 평가할 모델
        param_grid=param_grid,      # 테스트할 파라미터 목록
        cv=5,                       # 5-Fold 교차검증 (3 ~ 5 ~ 10)
        scoring='f1',               # F1-Score 기준으로 최적 파라미터 탐색 (f1, accuracy, precision, recall)
        n_jobs=-1,                  # 모든 CPU 코어 사용 (-1 = 최대 코어 사용, 1 = 1개 코어 사용)
        verbose=1                   # 진행상황 자세히 출력 (0=숨김, 1=간단, 2=상세)
    )
    
    grid_search.fit(X_train, y_train)
    
    # 최적 파라미터를 변수에 대입
    g_n_estimators = grid_search.best_params_['n_estimators']
    g_max_depth = grid_search.best_params_['max_depth']
    g_learning_rate = grid_search.best_params_['learning_rate']
    
    print("\n" + "="*60)
    print(f"최적 파라미터 발견 및 적용:")
    print(f"  n_estimators: {g_n_estimators}")
    print(f"  max_depth: {g_max_depth}")
    print(f"  learning_rate: {g_learning_rate}")
    print(f"최적 F1-Score (CV): {grid_search.best_score_:.4f}")
    print("="*60 + "\n")

# 학습 (변수 값 사용)
print(f"모델 학습: n_estimators={g_n_estimators}, max_depth={g_max_depth}, learning_rate={g_learning_rate}")

model = XGBClassifier(
    n_estimators=g_n_estimators,
    max_depth=g_max_depth,
    learning_rate=g_learning_rate,
    tree_method='hist',
    # device='cuda',          # GPU 사용
    random_state=42
)
model.fit(X_train, y_train)

# 예측
start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()

per_sample_time = (end_time - start_time) / len(X_test)

# 평가
print("정확도:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"샘플당 예측 시간: {per_sample_time:.6f}초")

# precision : 정밀도 (예측한 것 중 정답 비율)
# recall : 재현율 (실제 정답 중 맞춘 비율)
# f1-score : 정밀도와 재현율의 조화 평균
# support : 실제 정답 데이터 개수 (샘플 개수)

# accuracy : 전체 정확도 (정답 맞춘 비율 0 ~ 1)
# macro avg : 클래스별로 평균 낸 값 (클래스 불균형에 영향 적음)
# weighted avg : support(표본수)로 가중평균 낸 값 (실제 데이터 분포 반영)