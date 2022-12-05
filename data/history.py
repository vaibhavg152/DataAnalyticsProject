import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

df_a = pd.read_csv('ALQ_I.csv')
df_d = pd.read_csv("DUQ_I.csv")
df_s = pd.read_csv('SMQ_I.csv')
mdf2 = pd.merge(left=df_a, right=df_d, left_on='SEQN', right_on="SEQN")
mdf1 = pd.merge(left=df_a, right=df_s, left_on='SEQN', right_on="SEQN")
mdf  = pd.merge(left=mdf1, right=df_d, left_on='SEQN', right_on="SEQN")

a = list(map(lambda x: -1 if (math.isnan(x) or x==999) else x, mdf.ALQ160.to_numpy()))
s = list(map(lambda x: -1 if (math.isnan(x) or x==999) else x, mdf.SMD650.to_numpy()))
plt.ylim(0, 60)
plt.xlim(0, 30)
plt.scatter(a, s)
#plt.show()

nan = [(col, mdf.shape[0]-mdf[col].isna().sum()) for col in mdf.columns]
# print(sorted(nan, key=lambda x: -x[1]))

cols = ['ALQ101']
cols.append('ALQ120Q')
cols.append('ALQ120U')
cols.append('ALQ151')
cols.append('ALQ130')
cols.append('ALQ141Q')
cols.append('ALQ141U')
cols.append('SMQ020')
cols.append('SMQ890')
cols.append('SMQ900')
cols.append('SMQ910')
cols.append('SMQ925')
cols.append('SMD030')
cols.append('SMQ040')
cols.append('SMQ895')
cols.append('SMQ930')
cols.append('SMQ935')

# for col in cols:
#	print(col, mdf.groupby(col)[col].count())
for col in cols:
	mdf.groupby(col)[col].count()
mdf.groupby('SMAQUEX2')['SMAQUEX2'].count()
nan = [(col, mdf.shape[0]-mdf[col].isna().sum()) for col in mdf.columns]

c, h, m, w = df_d.DUQ250, df_d.DUQ290, df_d.DUQ330, df_d.DUQ200

hn = h.dropna()
mn = m.dropna()
cn = c.dropna()
plt.scatter(hn, mn)
# plt.show()

print((hn==mn).sum(), hn.shape[0])
print((hn==cn).sum(), hn.shape[0])
print((mn==cn).sum(), mn.shape[0])
# print(h.isna()==w.isna())
print((h==w).sum(), h.shape[0])

from sklearn.metrics import confusion_matrix as conf

print(conf(hn, mn))

labels = ["-1", "1", "2", "7", "9"]
print("\t".join(labels))
print(conf(h.fillna(-1), w.fillna(-1)))
print(conf(c.fillna(-1), w.fillna(-1)))
c = mdf.DUQ250.fillna(-1)
cig = mdf.SMQ925.fillna(-1)
print(conf(c, cig))

import statsmodels.api as sm
from scipy import stats


X = mdf[['SMQ930', 'DUQ210']]