import pandas as pd

train = pd.read_csv("C:/Users/chuan/OneDrive/Documents/Python Datasets/titanic/train.csv")
test = pd.read_csv("C:/Users/chuan/OneDrive/Documents/Python Datasets/titanic/test.csv")

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

print(train.columns)

def bar_chart(feature):
    survived = train[feature][train['Survived'] == 1].value_counts();
    print(survived)
    died = train[feature][train['Survived'] == 0].value_counts();
    df = pd.DataFrame([survived, died])
    df.index = ['Survived', 'Died']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))
    plt.show()

print(train.Name[0])
for dataset in [train]:
    print(dataset)
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in [train]:
    dataset['Title'] = dataset['Title'].map(title_mapping)

train.drop('Name', axis=1, inplace=True)

sex_mapping = {'male':0, 'female':1}
for dataset in [train]:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)


facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()


for dataset in [train]:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4



plt.show()
