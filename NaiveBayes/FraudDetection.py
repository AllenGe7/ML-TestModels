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

df["Time_Hr"] = df["Time"]/3600 # convert to hours


#plt.hist(df.Amount[df.Class==0],bins=50,color='g',alpha=0.5)

plt.yscale('log') # to see the tails

df['scaled_Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df = df.drop(['Time'],axis=1)

df = df.drop(['Amount'],axis=1)

gs =gridspec.GridSpec(2,1)
plt.figure(figsize=(6, 28*4))
for i, col in enumerate(df[df.iloc[:,0:2].columns]):
    plt.subplot(gs[i])
    sns.distplot(df[col][df.Class == 1], bins=50, color='r')
    sns.distplot(df[col][df.Class == 0], bins=50, color='g')
    plt.xlabel('')
    plt.title('feature: ' + str(col))



def split_data(df, drop_list):
    df = df.drop(drop_list, axis = 1)
    print(df.columns)
    from sklearn.model_selection import train_test_split
    y = df['Class'].values
    X = df.drop(['Class'], axis = 1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y)
    print("train-set size ", len(y_train), "\ntest-set size: ", len(y_test))
    print("fraud case in test-set", sum(y_test))
    return X_train, X_test, y_train, y_test

def get_predictions(clf, X_train, y_train, X_test):
    clf = clf
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)
    train_pred = clf.predict(X_train)
    print('train-set confusion matrix:\n', confusion_matrix(y_train, train_pred))
    return y_pred, y_pred_prob

def print_scores(y_test,y_pred,y_pred_prob):
    print('test-set confusion matrix:\n', confusion_matrix(y_test,y_pred))
    print("recall score: ", recall_score(y_test,y_pred))
    print("precision score: ", precision_score(y_test,y_pred))
    print("f1 score: ", f1_score(y_test,y_pred))
    print("accuracy score: ", accuracy_score(y_test,y_pred))
    print("ROC AUC: {}".format(roc_auc_score(y_test, y_pred_prob[:,1])))

drop_list = []
X_train, X_test, y_train, y_test = split_data(df, drop_list)
y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)
print_scores(y_test,y_pred,y_pred_prob)