# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:57:43 2018

@author: pannnit
"""
## Libraries
import pandas as pd
import chardet
import numpy as np
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score



with open(r'C:\Users\Elastomer_py_new.csv', 'rb') as f:
    result = chardet.detect(f.readline())  # or readline if the file is large


Mot= pd.read_csv(r'C:\Users\Elastomer_py_new.csv', encoding=result['encoding'])
Mot.head(1)
list(Mot)
Mot['Average of Chlorides'].max()
Mot['Average of Average Motor Diff Pressure'].max()

k1 = Mot.loc[:, ['Location', 'Motor','StatorElastomer','DisAssyCause','Conventional Elastomer Condition', 'Primary Failure P/N','Tool Size','Average of Average Motor Diff Pressure','Sum of DistanceDrilled','Average of Chlorides','Average of Solid Content']]
import seaborn as sns
# count the number of NaN values in each column
print(k1.isnull().sum())
k1.fillna(0)
sns.boxplot(x=k1['Average of Chlorides'])
sns.boxplot(x=k1['Average of Average Motor Diff Pressure'])
Chl = k1['Average of Chlorides'] > 100000
k1.loc[Chl, 'Average of Chlorides'] = 0
psi = k1['Average of Average Motor Diff Pressure'] > 200
k1.loc[psi, 'Average of Average Motor Diff Pressure'] = 200
### Marking suspensions and Failures
k1['Conventional Elastomer Condition'] = k1['Conventional Elastomer Condition'].str.replace('Reline: Shelf Life Expired','Shelf Life Expired')
k1['Status']  = np.where(k1['Conventional Elastomer Condition'].str.contains('(?:OK|Shelf Life Expired)',
                                        regex=True),
                   'S', 'F')
## Removing Special Characters
k1['Primary Failure P/N'] = k1['Primary Failure P/N'].str.replace('Elastomer','0')
k1['Primary Failure P/N'] = k1['Primary Failure P/N'].str.replace('H','1')
k1['Primary Failure P/N'] = k1['Primary Failure P/N'].str.replace('N','2')

k1=k1.replace('\>','',regex=True).astype(str)
k1=k1.replace('\<','',regex=True).astype(str)


## Decision Tree

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in k1.columns:
        if k1[column_name].dtype == object:
            k1[column_name] = le.fit_transform(k1[column_name])
        else:
            pass
A = k1.drop('Status', axis=1)
B = A.drop('Conventional Elastomer Condition', axis=1)
X = B.drop('DisAssyCause', axis=1)
y = k1['Status']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 100)
for depth in range(1,8):
 tree_classifier = tree.DecisionTreeClassifier(
  max_depth=depth, random_state=0)
 if tree_classifier.fit(X_train,y_train).tree_.max_depth < depth:
  break
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
crossvalidation = KFold(n=X_train.shape[0], n_folds=8,
 shuffle=True, random_state=1)
score = np.mean(cross_val_score(tree_classifier, X_train, y_train,
  scoring='accuracy', cv=crossvalidation, n_jobs=1))
print ("Depth: %i Accuracy: %.3f" % (depth,score))
## For the graph
## Tree with Gini IndexPython

clf_gini = tree.DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=7, min_samples_leaf=10)
clf_gini.fit(X_train, y_train)

#Decision Tree with Information Entropy 

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=7, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

# evaluate algorithm

y_pred = clf_gini.predict(X_test)
y_pred_en = clf_entropy.predict(X_test)
#Accuracy


print ("Accuracy for gini", accuracy_score(y_test,y_pred)*100)
print ("Accuracy for entropy", accuracy_score(y_test,y_pred_en)*100)
import graphviz
list(X)
tree.export_graphviz(clf_gini,out_file='tree.dot')
from sklearn.externals.six import StringIO  
import pydot 
dot_data = StringIO() 
tree.export_graphviz(clf_gini, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph[0].write_pdf("Elastomer.pdf")
tree.export_graphviz(clf_entropy, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph[0].write_pdf("Elastomer_Entropy.pdf")


