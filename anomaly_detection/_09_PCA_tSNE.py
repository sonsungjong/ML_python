# PCA 와 t-SNE
# 시각화 기반의 이상탐지 기법
# PCA 진행 전에는 반드시 정규화를 거쳐야한다

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

iris = load_iris()
df = pd.DataFrame(data=np.c_[iris.data, iris.target], columns=['sepal length', 'sepal width','petal length','petal width','target'])

df.head()

# 스케일러 생성
scaler = StandardScaler()

# PCA 객체 생성
pca = PCA()

# 파이프라인 생성
pipeline = make_pipeline(scaler, pca)
pipeline.fit(df.drop(['target'], axis=1))

# pca.n_components (차원 축소 주성분 갯수)
print(pca.n_components_)
features = range(pca.n_components_)
feature_df = pd.DataFrame(data=features, columns=['pc_feature'])

# pca.explained_variance_ratio_ (설명력)
variance_df = pd.DataFrame(data=pca.explained_variance_ratio_, columns=['variance'])
pc_feature_df = pd.concat([feature_df, variance_df], axis=1)
print(pc_feature_df)

x = df.drop(['target'], axis=1).reset_index(drop=True)
y = df['target'].reset_index(drop=True).astype(str)

# 정규화
X_ = StandardScaler().fit_transform(x)

# 2개의 주성분으로 차원 축소
pca = PCA(n_components=2)
pc = pca.fit_transform(X_)

pc_df = pd.DataFrame(pc, columns=['PC1','PC2']).reset_index(drop=True)
pc_df=pd.concat([pc_df, y], axis=1)

plt.style.use(['dark_background'])
plt.rcParams['figure.figsize'] = [10, 10]
sns.scatterplot(data=pc_df, x='PC1', y='PC2', hue=y, legend='brief', s=50, linewidth=0.5);

# 각 클러스터 중심에서 멀리 떨어진 Data 일수록 이상치
plt.show()