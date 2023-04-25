# 마할라노비스 거리 이상 탐지
# Mahalanobis Distance (MD) 를 사용하여 점과 분포 사이의 거리를 찾는 방법
# 중심점까지의 거리를 기준으로 이상치를 식별하는 방법
# 다변량 데이터에서 매우 효과적으로 동작

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 마할라노비스
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=30, n_features=2, centers=1, random_state=1)

X[0,0] = 10
X[0,1] = -10

# 하이퍼파라미터 0.2
outlier_detector = EllipticEnvelope(contamination=0.2)
# outlier_detector = EllipticEnvelope(contamination=0.05)           # 더 엄격하게

outlier_detector.fit(X)

pred = outlier_detector.predict(X)

print(pred)

# Raw Data + Pred Data
df = pd.DataFrame(X, columns=['col1', 'col2'])
df['outlier'] = pred
print(df.head())

plt.style.use(['dark_background'])
sns.scatterplot(x='col1', y='col2', hue='outlier', data=df)
plt.show()          # 주피터노트북에선 %matplotlib inline
