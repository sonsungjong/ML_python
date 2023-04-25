import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# %matplotlib inline            # vscode에서는 plt.show()를 통해서 그래프 보기 -- 주피터노트북 전용임

# 3-sigma Rule 이상탐지

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

df = pd.DataFrame({
    "name":["KATE", "LOUISE", "JANE", "JASON", "TOM", "JACK"],
    "weight":["59", "61", "55", "66", "52", "110"],
    "height":["120", "123","115","145","64","20"]
})

print(df)
print(df.info())

# 숫자로 형변환
df['weight'] = df['weight'].astype(int)
df['height'] = df['height'].astype(int)

# 평균과 표준편차를 구해 기준을 정의하는 기법을 사용하자 (정규성 검증 필요)
df['UCL_W'] = df['weight'].mean() + 2*df['weight'].std()
df['LCL_W'] = df['weight'].mean() - 2*df['weight'].std()

df['UCL_H'] = df['height'].mean() + 2*df['height'].std()
df['LCL_H'] = df['height'].mean() - 2*df['height'].std()

print(df)

plt.style.use(['dark_background'])

sns.scatterplot(x=df['name'], y=df['weight']);
plt.axhline(y=df['UCL_W'][0], color='r', linewidth=1)
plt.axhline(y=df['LCL_W'][0], color='r', linewidth=1)
plt.gcf().set_size_inches(15, 5)
plt.show()

sns.scatterplot(x=df['name'], y=df['height']);
plt.axhline(y=df['UCL_H'][0], color='r', linewidth=1)
plt.axhline(y=df['LCL_H'][0], color='r', linewidth=1)
plt.gcf().set_size_inches(15,5)
plt.show()