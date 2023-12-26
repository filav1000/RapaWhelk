import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv('/home/user/filav/IRR/ML model/rapa_age_sex_GFCM.csv') # над 4000 бройки
print(len(df))

# drop all NA rows
df.dropna(inplace=True)
print(len(df))
print(df.describe())

print(df['Sex'].value_counts())
print(df.sort_values('BW', ascending=True).head(5))

ax = sns.boxplot(data=df, y='BW', x='Sex')
plt.show()

sns.scatterplot(data = df, x = "BW", y = "W", hue = "Sex", s =3)
#plt.xscale("log")
#plt.yscale("log")
plt.show()

#unique, counts = np.unique(df["Sex"].values, return_counts=True)
#plt.bar(unique, counts)
#plt.title("No of samples by sex")
#plt.xlabel("Sex")
#plt.ylabel("Counts")
#plt.show()

sns.countplot(data = df, x = "Sex", hue = "Sex")
plt.show()

df["Sex"] = [1 if x == "M" else 0 for x in df["Sex"]]
print(df.head())
#df["AGE"] = df["AGE"].astype("float")

X = df[["L", "W", "BW", "CW"]].values
y = df.iloc[:, -1].values

# импортвам imblearn и след това oversampler
import imblearn
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(sampling_strategy = 0.8)  # евентуално може да се даде random_state = 123, 0.8 е съотношението мъжки : женски
X_resampled, y_resampled = ros.fit_resample(X, y)  # фитвам данните

# проверка за oversampling
print("Женските са {}, мъжките {}".format(np.unique(y_resampled, return_counts=True)[1][0], np.unique(y_resampled, return_counts=True)[1][1]))
print(np.unique(y, return_counts=True))

unique, counts = np.unique(y_resampled, return_counts = True)
df_resampled = pd.DataFrame({'Sex': unique, 'Counts': counts})

sns.barplot(data = df_resampled, x = 'Sex', y = 'Counts', hue = 'Sex')
plt.xticks([0,1], labels = ["F", "M"])
plt.legend(["F", "M"])
plt.show()

