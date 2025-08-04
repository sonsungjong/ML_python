import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. 데이터 로드 및 분할
df = pd.read_csv('mdi.csv')
target_col = 'HUMAN'
X = df.drop(columns=[target_col])
y = df[target_col]

# 2. 모델 학습
model = RandomForestClassifier(
    n_estimators=30,            # 30 ~ 50
    max_depth=5,                # 5 ~ 10
    random_state=42
)
model.fit(X, y)

# 3. 모델 저장
joblib.dump(model, 'lzr_ai_model.joblib')

print('모델 저장 완료')
