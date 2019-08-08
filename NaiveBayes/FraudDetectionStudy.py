import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,auc,roc_auc_score
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


df = pd.read_csv(r"C:\Users\chuan\OneDrive\Documents\Python Datasets\creditcardfraud\creditcard.csv")

df['Time_Hour'] = df['Time']/3600

df = df.drop(['Time'], axis = 1)


'''
fig,(graph1, graph2) = plt.subplots(2, 1, sharex = True, figsize = (10,5))
graph1.hist(df.Amount[df.Class==0], color = 'b', bins = 50)
graph1.set_yscale('log')
graph2.hist(df.Amount[df.Class==1], color = 'r', bins = 50)
plt.yscale('log')
plt.show()


gs = gridspec.GridSpec(10, 1)
plt.figure(figsize=(6,(28*4)))
for i, col in enumerate(df.iloc[:,11:20]):
    plt.subplot(gs[i])
    sns.distplot(df[col][df.Class==1], color = 'g', bins = 50)
    sns.distplot(df[col][df.Class==0], color='r', bins=50)
    plt.xlabel(col)
plt.show()
'''
def splitdata(df, drop_list):
    df = df.drop(drop_list, axis = 1)

    from sklearn.model_selection import train_test_split
    y = df['Class'].values
    X = df.drop(['Class'], axis = 1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 42, test_size = 0.2, stratify = y)
    print("train data size ", len(y_train))
    print("train test size", len(y_test))
    return X_train, X_test, y_train, y_test

def testData(clf, X_train, X_test, y_train):
    clf = clf
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)

    train_pred = clf.predict(X_train)
    print(confusion_matrix(y_train, train_pred))
    return y_pred, y_pred_prob
def print_scores(y_test,y_pred,y_pred_prob):
    print('test-set confusion matrix:\n', confusion_matrix(y_test,y_pred))
    print("recall score: ", recall_score(y_test,y_pred))
    print("precision score: ", precision_score(y_test,y_pred))
    print("f1 score: ", f1_score(y_test,y_pred))
    print("accuracy score: ", accuracy_score(y_test,y_pred))
    print("ROC AUC: {}".format(roc_auc_score(y_test, y_pred_prob[:,1])))
drop_list = []
X_train, X_test, y_train, y_test = splitdata(df, drop_list)
y_pred, y_pred_prob = testData(GaussianNB(), X_train, X_test, y_train)
print_scores(y_test,y_pred,y_pred_prob)



