#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Sesha Sai Sreevani Kappagantula
# N11264916
# ssk785

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import export_graphviz


#X,y values initialization
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = df.iloc[50:150, 4].values
y = np.where(y == 'Iris-versicolor', -1, 1)
X = df.iloc[50:150, [0,1,2,3]].values

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#train and predict using Decisiontree classifier for all 4 features with gini criterion
lr = tree.DecisionTreeClassifier(criterion='gini',max_features=4)
lr.fit(X_train, y_train)
y_pred=lr.predict(X_test)

#metrics valuation
print('\nMetrics when gini criterion is used:\n')
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Logloss:',log_loss(y_test,y_pred))

#train and predict using Decision tree classifier for all 4 features with entropy criterion
from sklearn.linear_model import LogisticRegression
lr = tree.DecisionTreeClassifier(criterion='entropy',max_features=4)
lr.fit(X_train, y_train)
y_pred=lr.predict(X_test)

#metrics valuation
print('\nMetrics when entropy criterion is used:\n')
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Logloss:',log_loss(y_test,y_pred))


# List of values to try for max_depth 1,6 depth range:
max_depth_range = list(range(1, 7))
# List to store the accuracy values for gini criterion for each value of max_depth:

print('\nAccuracy for each depth value when gini criterion is used:')
for depth in max_depth_range:
    clf = tree.DecisionTreeClassifier(max_depth = depth,criterion='gini',max_features=4 
                             )
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('\ndepth=%d'%depth)
    print('score=%.2f'%score)

# List to store the accuracy values for entropy criterion for each value of max_depth:
print('\nAccuracy for each depth value when entropy criterion is used:')
for depth in max_depth_range:
    
    clf = tree.DecisionTreeClassifier(max_depth = depth,criterion='entropy',max_features=4 
                             )
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('\ndepth=%d'%depth)
    print('score=%.2f'%score)

#code for decision tree plotting for iris data 

# Importing Libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load data and store it into pandas DataFrame objects
iris = load_iris()
X = pd.DataFrame(iris.data[:, :], columns = iris.feature_names[:])
y = pd.DataFrame(iris.target, columns =["Species"])

#fitting a DecisionTreeClassifier for 6 depth values in 'gini' criterion and plotting .dot files in each case
max_depth_range = list(range(1, 7))
names=['depth1','depth2','depth3','depth4','depth5','depth6']
for i in max_depth_range:
    tree = DecisionTreeClassifier(max_depth = i,criterion='gini')
    tree.fit(X,y)
# Visualize Decision Tree
    from sklearn.tree import export_graphviz

# Creates dot file named tree.dot
    export_graphviz(
            tree,
            out_file =  "Tree_gini_" + names.pop(0)+".dot",
            feature_names = list(X.columns),
            class_names = iris.target_names,
            filled = True,
            rounded = True)

    
#fitting a DecisionTreeClassifier for 6 depth values in 'entropy' criterion and plotting .dot files in each case
max_depth_range = list(range(1, 7))
names=['depth1','depth2','depth3','depth4','depth5','depth6']
for i in max_depth_range:
    tree = DecisionTreeClassifier(max_depth = i,criterion='entropy')
    tree.fit(X,y)
# Visualize Decision Tree
    from sklearn.tree import export_graphviz

# Creates dot file named tree.dot
    export_graphviz(
            tree,
            out_file =  "Tree_entropy_" + names.pop(0)+".dot",
            feature_names = list(X.columns),
            class_names = iris.target_names,
            filled = True,
            rounded = True)


print('\n12 .dot files will be downloaded in Downloads.')
print('Please open the .dot files to see the decision tree plots for each of the 12 cases taken above')


# In[ ]:


'''Conclusion:
        -Overall, Gini impurity criterion is giving higher accuracy and lower logloss value than entropy criterion.
         
        -We also notice that with the increase in depth, accuracy increases but after certain count the
         accuracy drops a little.This is because of overfitting.When too many depths are considered, overfitting
         occurs leading to drop in accuracy or remains unchanged
'''


# In[ ]:


'''Analysis of depth vs accuracy:

    Depths           Accuracy when gini      Accuracy when entropy

    Depth 1:               0.8666                 0.8666
    Depth 2:               0.8333                 0.8333
    Depth 3:               0.8666                 0.9
    Depth 4:               0.9333                 0.9
    Depth 5:               0.9333                 0.9333
    Depth 6:               0.9                    0.9333
    '''


# In[ ]:


'''***Note:: This code when run also produces 12 .dot files (6 files for each of the criterion used.
   When the downloaded .dot files are opened we can see the decision tree plot for each case)
   Also, if you are only able to view the code on opening the .dot file due to issue with graphviz software on your PC,
   please convert the .dot file to png to see the correct depiction of decision tree plot'''

