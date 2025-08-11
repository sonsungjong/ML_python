# pip install scikit-learn
# pip install lightgbm
# pip install xgboost
# pip install catboost

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib

select_model_number = 2

model_name = {
    1: "RandomForest",
    2: "LightGBM",
    3: "Pytorch",
    4: "XGBoost",
    5: "CatBoost"
}


# 1. 데이터 로드 및 분할
df = pd.read_csv('mdi.csv', sep=",")
target_col = 'HUMAN'
X = df.drop(columns=[target_col])
y = df[target_col]

# 2. 모델 선택
model = None
if model_name[select_model_number] == "RandomForest":
    model = RandomForestClassifier(
        n_estimators=30,            # 30 ~ 50
        max_depth=5,                # 5 ~ 10
        random_state=42
    )
elif model_name[select_model_number] == "LightGBM":
    model = LGBMClassifier(
        n_estimators=30,            # 30 ~ 100
        max_depth=5,                # 3 ~ 10
        n_jobs=1,           # CPU 코어 1개만 사용
        random_state=42
    )
elif model_name[select_model_number] == "Pytorch":
    pass
elif model_name[select_model_number] == "XGBoost":
    model = XGBClassifier(
        n_estimators=30,            # 30 ~ 100
        max_depth=5,                # 3 ~ 10
        eval_metric='logloss',
        n_jobs=1,           # CPU 코어 1개만 사용
        tree_method='hist',
        random_state=42
    )
elif model_name[select_model_number] == "CatBoost":
    model = CatBoostClassifier(
        iterations=30,   # 트리 개수 (n_estimators)
        depth=5,         # 트리 깊이 (max_depth)
        verbose=0,       # 로그 숨김
        random_seed=42,
        allow_writing_files=False       # 로그 생성 안함
    )

if model is not None:
    # 3. 모델 학습
    model.fit(X, y)
    # 4. 모델 저장
    joblib.dump(model, 'lzr_ai_model.joblib')
    print('모델 저장 완료')
