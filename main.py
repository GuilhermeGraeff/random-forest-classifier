import pandas as pd

DROP_CAPITAL_GAIN_AND_LOSS = False

import itertools
from time import time
import math

# Sklearn

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier


from sklearn import preprocessing as preprocessing

# Metrics
from sklearn.metrics import recall_score as rec
from sklearn.metrics import precision_score as prec
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import f1_score as f1_scr

# Drive
# from google.colab import drive

def round_up(n, decimals=0):
  multiplier = 10 ** decimals
  return math.ceil(n * multiplier) / multiplier

DataSet = pd.read_csv('adult.csv')
DataSet_test = pd.read_csv('adult_test.csv')
DataSet = DataSet.append(DataSet_test, ignore_index=True)



print(DataSet)
print(DataSet)
label_encoder=preprocessing.LabelEncoder()

quantitative_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

mode = DataSet.mode()


for feature in quantitative_features:
    DataSet = DataSet.replace("?",mode[feature][0])
    label_encoder.fit(DataSet[feature])
    DataSet[feature]=label_encoder.transform(DataSet[feature])

print(DataSet['outcome'].unique())
DataSet = DataSet.replace({ 'outcome' : {
    " <=50K " :  " <=50K" 
  }
})

DataSet = DataSet.replace({ 'outcome' : {
    " <=50K." :  " <=50K" 
  }
})
DataSet = DataSet.replace({ 'outcome' : {
    " >50K." :  " <=50K" 
  }
})

print(DataSet['outcome'].unique())

label_encoder.fit(DataSet['outcome'])
DataSet['outcome']=label_encoder.transform(DataSet['outcome'])

print(DataSet['outcome'].unique())

for i in DataSet.index: 
    if DataSet["outcome"][i] == 2:
        DataSet["outcome"][i] = 1

print(DataSet['outcome'].unique())

print(DataSet)

print("Outcome: 0 stands for <=50k, 2 stands for >50k")

print("Capital-gain has "+str((DataSet['capital-gain']==0).sum()/(DataSet['capital-gain'].count())*100) + " % of it's values as 0")
print("Capital-loss has "+str((DataSet['capital-loss']==0).sum()/(DataSet['capital-loss'].count())*100) + " % of it's values as 0")

DataSet.drop(columns=['education'], inplace=True)

if DROP_CAPITAL_GAIN_AND_LOSS:
  DataSet.drop(columns=['capital-gain', 'capital-loss'], inplace=True)
  print("Capital-loss and Capital-gainhas been dropped")

# print(DataSet)

X = DataSet.drop(columns=['outcome'])
y = DataSet['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clfRandomForest = RandomForestClassifier(criterion='entropy', max_depth=None, max_features='log2',min_samples_leaf = 2, min_samples_split = 5, n_estimators = 800, random_state = 128 )
clfRandomForest.fit(X_train, y_train)
y_hat = clfRandomForest.predict(X_test)

print(f'Scale: Accuracy: {str(acc(y_test,y_hat))} Precision: {str(prec(y_test,y_hat))} Recal: {str(rec(y_test,y_hat))} F1_Score: {str(f1_scr(y_test,y_hat))} ')