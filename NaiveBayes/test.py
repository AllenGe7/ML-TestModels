import pandas as pd
import matplotlib.pyplot as plt


df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
                  index=['cobra', 'viper', 'sidewinder'],
                  columns=['max_speed', 'shield'])

print(df.loc[['cobra', 'viper']])

train = pd.read_csv("C:/Users/chuan/OneDrive/Documents/Python Datasets/titanic/train.csv")

mapping = {'male':1, 'female':2}

for test in [train]:
    train.Sex = test['Sex'].map(mapping)

for dataset in [train]:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1
train['Age'].fillna(train.Age.mean(), inplace=True)

print(train.Cabin.value_counts())

for dataset in [train]:
    dataset['Cabin'] = dataset['Cabin'].str[0]

Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))

plt.show()
#for test in train.itertuples():

