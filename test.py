import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

data = pd.read_csv(r"C:\Users\王崇\Desktop\test.csv")
data.columns[37:45]
data.sort_values(by='year', inplace=True)

for col in data.columns[2:]:
    if col in data.columns[37:45]:
        continue
    q1 = data[col].quantile(0.25)
    q2 = data[col].quantile(0.75)
    mean = data[col].mean()
    data[col] = np.where((data[col]<q1) | (data[col]>q2), mean, data[col])


X_train = data[data['year'] <= 2016].iloc[:, 2:]
y_train = data[data['year'] <= 2016].iloc[:, 1]
X_test = data[data['year'] > 2016].iloc[:, 2:]
y_test = data[data['year'] > 2016].iloc[:, 1]
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

y_train.value_counts()

X_train.mean()
X_train.describe()


#异常值检测
outlier = OneClassSVM(gamma='auto').fit(X_train)
temp = outlier.predict(X_train)
np.where(temp==-1)[0].__len__()
X_train = X_train.iloc[np.where(temp==1)]
y_train = y_train.iloc[np.where(temp==1)]


clf = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.8)),
                ('tree', DecisionTreeClassifier(max_depth=7, class_weight='balanced'))])
# clf = Pipeline([('scaler', StandardScaler()), ('select', SelectKBest(k=20)),
#                 ('tree', DecisionTreeClassifier(max_depth=8, class_weight='balanced'))])

clf.fit(X_train, y_train)
roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
accuracy_score(y_train, clf.predict(X_train))

y_pred = clf.predict(X_test)
roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
accuracy_score(y_test, y_pred)
clf['tree'].get_depth()


clf = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.9)),
                ('svm', SVC(probability=True, class_weight='balanced'))])

clf.predict_proba(X_test)
y_pred[:10]
confusion_matrix(y_test, y_pred)
recall_score(y_test, y_pred)


# np.where(y_test == 0)[0].__len__()
# np.count_nonzero(y_test)
