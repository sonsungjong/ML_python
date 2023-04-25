import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 3-sigma Rule 이상탐지
# Box Plot 이용

plt.style.use(['default'])

test = pd.DataFrame([1, 5, 9, 10, 15, 20, 34])
fig, ax = plt.subplots(1, 1, figsize=(4,4))
a = list(test[0])
box = ax.boxplot(a);
plt.show()

data_a = np.random.normal(0, 2.0, 1000)
data_b = np.random.normal(-3.0, 1.5, 500)
data_c = np.random.normal(1.2, 1.5, 1500)

labels = ['data_a', 'data_b', 'data_c']

fig, ax = plt.subplots()

box = ax.boxplot([data_a, data_b, data_c], whis=1.5);
ax.set_ylim(-10.0, 10.0);
ax.set_xlabel('Data Type');
ax.set_ylabel('Value');
plt.show()

# Box plot과 분포를 같이 확인해야할 때 (야매)
data_a_df = pd.DataFrame(data_a)
data_a_df.boxplot()

for i, d in enumerate(data_a_df):
    y = data_a_df[d]
    x = np.random.normal(i + 1, 0.04, len(y))
    plt.scatter(x,y)
plt.title("boxplot with scatter", fontsize=20)
plt.show()

# Outlier 검출하기
def getBoxPlotData(labels, bp):
    rows_list = []
    for i in range(len(labels)):
        dict1 = {}
        dict1['label'] = labels[i]
        dict1['lower_whisker'] = bp['whiskers'][i*2].get_ydata()[1]
        dict1['lower_quartile'] = bp['boxes'][i].get_ydata()[1]
        dict1['median'] = bp['medians'][i].get_ydata()[1]
        dict1['upper_quartile'] = bp['boxes'][i].get_ydata()[2]
        dict1['upper_whisker'] = bp['whiskers'][(i*2)+1].get_ydata()[1]
        dict1['Outlier'] = bp['fliers'][i].get_ydata()
        rows_list.append(dict1)
    return pd.DataFrame(rows_list)

print(getBoxPlotData(labels, box))

fliers = [item.get_ydata() for item in box['fliers']]
print(fliers)

print(fliers[0])            # data_a
print(fliers[1])            # data_b
print(fliers[2])            # data_c
