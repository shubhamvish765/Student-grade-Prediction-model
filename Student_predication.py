from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("student-mat.csv")


print('Total number of students: ',len(df))
print("Parameter are: ",df.columns)
df.info()
df.describe()
df['G3'].describe()
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

sns.countplot(x="age",data=df)


df['G3'].describe()

df=df.dropna()
print((df.isnull()).sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in df.columns:
 df[col]=le.fit_transform(df[col])

  df.head()

df.corr()['G3'].sort_values()

# df = df.drop(['school', 'G1', 'G2'], axis='columns')

most_correlated = df.corr().abs()['G3'].sort_values(ascending=False)
most_correlated = most_correlated[:9]
most_correlated
df = df.loc[:, most_correlated.index]
df.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, df['G3'], test_size = 0.3, random_state=0)

X_train.head()


X_train = X_train.drop('G3', axis='columns')
X_test = X_test.drop('G3', axis='columns')
lr = LinearRegression()
model = lr.fit(X_train, y_train)

print(lr.score(X_test, y_test))

predictions = lr.predict(X_test)

plt.plot(y_test, predictions, 'o')
m, b = np.polyfit(y_test,predictions, 1)
plt.plot(y_test, m*y_test + b)
plt.xlabel("Actual Grade")
plt.ylabel("Predicted Grade")

