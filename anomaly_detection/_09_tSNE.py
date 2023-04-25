from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

# 데이터셋 로드
iris = load_iris()
df = pd.DataFrame(data=np.c_[iris.data, iris.target], columns=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])

df.head()

# class target 정보 제외
train_df = df[['sepal length', 'sepal width', 'petal length', 'petal width']]

# 2차원 t-SNE 임베딩
# n_components : 차원수
# perplexity : 주변 이웃 데이터 참조 갯수
# learning_rate : 학습률, 크면 너무 붙어있게 되고 작으면 너무 멀어짐 (특징X)
# n_iter : 최적화 반복 횟수
tsne_np = TSNE(n_components = 2).fit_transform(train_df)

# numpy array -> DataFrame 변환
tsne_df = pd.DataFrame(tsne_np, columns = ['component 0', 'component 1'])
print(tsne_df.head())

tsne_df['target'] = df['target']

tsne_df_0 = tsne_df[tsne_df['target'] == 0]
tsne_df_1 = tsne_df[tsne_df['target'] == 1]
tsne_df_2 = tsne_df[tsne_df['target'] == 2]

# target 별 시각화
plt.scatter(tsne_df_0['component 0'], tsne_df_0['component 1'], color='pink', label='setosa')
plt.scatter(tsne_df_1['component 0'], tsne_df_1['component 1'], color='purple', label='versicolor')
plt.scatter(tsne_df_2['component 0'], tsne_df_2['component 1'], color='yellow', label='virginica')

plt.xlabel('component 0')
plt.ylabel('component 1')
plt.legend()
# plt.show()

# 3차원 t_SNE 임베딩
tsne_np = TSNE(n_components= 3).fit_transform(train_df)

# numpy -> pandas 변환
tsne_df = pd.DataFrame(tsne_np, columns = ['component 0', 'component 1', 'component 2'])
print(tsne_df.head())

# 3차원 그래프 셋팅
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

# class target 정보 불러오기
tsne_df['target'] = df['target']

# target 별 분리
tsne_df_0 = tsne_df[tsne_df['target'] == 0]
tsne_df_1 = tsne_df[tsne_df['target'] == 1]
tsne_df_2 = tsne_df[tsne_df['target'] == 2]

# target 별 시각화
ax.scatter(tsne_df_0['component 0'], tsne_df_0['component 1'], tsne_df_0['component 2'], color='pink', label='setosa')
ax.scatter(tsne_df_1['component 0'], tsne_df_1['component 1'], tsne_df_1['component 2'], color='purple', label='versicolor')
ax.scatter(tsne_df_2['component 0'], tsne_df_2['component 1'], tsne_df_2['component 2'], color='yellow', label='virginica')

ax.set_xlabel('component 0')
ax.set_ylabel('component 1')
ax.set_zlabel('component 2')
ax.legend()
plt.show()
