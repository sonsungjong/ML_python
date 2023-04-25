# Robust Random Cut Forest
# pip install rrcf

import numpy as np
import pandas as pd
import rrcf
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

# 데이터 파라미터 셋팅
np.random.seed(0)
n = 2010
d = 3

# 하이퍼 파라미터 결정
num_trees = 100
tree_size = 256

# 샘플 데이터
X = np.zeros((n,d))
X[:1000, 0] = 5
X[1000:2000, 0] = -5
X += 0.01 * np.random.randn(*X.shape)
size=(n//tree_size, tree_size)

print(X)

# Construct forest
forest = []
while len(forest) < num_trees:
    # Select random subsets of points uniformly from point set
    ixs = np.random.choice(n, size=(n//tree_size, tree_size), replace=False)
    # Add sampled trees to forest
    trees = [rrcf.RCTree(X[ix], index_labels=ix) for ix in ixs]
    forest.extend(trees)

# Compute average CoDisp
avg_codisp = pd.Series(0.0, index=np.arange(n))
index = np.zeros(n)
for tree in forest:
    codisp = pd.Series({leaf : tree.codisp(leaf) for leaf in tree.leaves})
    avg_codisp[codisp.index] += codisp
    np.add.at(index, codisp.index.values, 1)
avg_codisp /= index

print(len(avg_codisp))
sns.displot(avg_codisp);
plt.show()

threshold = avg_codisp.nlargest(n=10).min()

fig = plt.figure(figsize=(12, 4.5))
ax = fig.add_subplot(121, projection='3d')
sc = ax.scatter(X[:,0], X[:,1], X[:,2], c=np.log(avg_codisp.sort_index().values), cmap='gnuplot2')
plt.title('log(CoDisp)')
ax = fig.add_subplot(122, projection='3d')
sc = ax.scatter(X[:,0], X[:,1], X[:,2], linewidths=0.1, edgecolors='k', c=(avg_codisp >= threshold).astype(float), cmap='cool')
plt.title('CoDisp above 99.5th percentile')
plt.show()

# 스트리밍 예제
n = 730
A = 50
center = 100
phi = 30
T = 2*np.pi / 100
t = np.arange(n)
sin = A*np.sin(T*t-phi*T) + center
sin[235:255] = 80

# 트리 파라미터 설정
num_trees = 40
shingle_size = 4
tree_size = 256

forest = []
for _ in range(num_trees):
    tree = rrcf.RCTree()
    forest.append(tree)

points = rrcf.shingle(sin, size=shingle_size)
avg_codisp = {}

for index, point in enumerate(points):
    for tree in forest:
        if len(tree.leaves) > tree_size:
            tree.forget_point(index - tree_size)
        tree.insert_point(point, index=index)
        new_codisp = tree.codisp(index)
        if not index in avg_codisp:
            avg_codisp[index] = 0
        avg_codisp[index] += new_codisp / num_trees

fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:red'
ax1.set_ylabel('Data', color=color, size=14)
ax1.plot(sin, color=color)
ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
ax1.set_ylim(0, 160)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('CoDisp', color=color, size=14)
ax2.plot(pd.Series(avg_codisp).sort_index(), color=color)
ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
ax2.grid('off')
ax2.set_ylim(0, 160)
plt.title('Sine wave with injected anomaly (red) and anomaly score (blue)', size=14)
plt.show()