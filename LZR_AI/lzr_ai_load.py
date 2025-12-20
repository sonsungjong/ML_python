# pip install scikit-learn
# pip install lightgbm
# pip install xgboost
# pip install catboost

# xgboost 버전

import joblib
import pandas as pd
import numpy as np
import time

# 1. 모델 로드 (1회만)
model = joblib.load('lzr_ai_model.joblib')

def predict_distance_array(distance_array_1096):
    arr = np.array(distance_array_1096)
    if arr.ndim != 1 or arr.shape[0] != 1096:
        raise ValueError("입력 데이터는 1차원, 길이 1096(4*274) 이어야 합니다.")
    arr = arr.reshape(1, -1)

    start_time = time.perf_counter()        # 시간 측정용
    pred = model.predict(arr)[0]
    end_time = time.perf_counter()

    print(f'TIME: {(end_time - start_time):.6f} seconds')

    return int(pred)

# 예시 입력 (실시간으로 들어오는 거리값 1096개)
test_sample = np.random.randint(0, 65001, size=1096)  # 예시: 0~65000 랜덤값

# 예측
pred = predict_distance_array(test_sample)
if pred == 1:
    print("HUMAN DETECTED")
else:
    print("NO HUMAN")