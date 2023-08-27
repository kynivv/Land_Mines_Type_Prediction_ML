# Libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acs

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC

# Data Import
dataset = pd.read_csv('Mine_Dataset.csv')
print(dataset)

# EDA
print(dataset.info())
print(dataset.isnull().sum())


# Train Test Split
features = dataset.drop('M', axis= 1)
target = dataset['M']

X_train, X_test, Y_train, Y_test = train_test_split(features, target,
                                                    shuffle= True,
                                                    test_size= 0.25,
                                                    random_state= 24
                                                    )


# Model Training
models = [AdaBoostClassifier(),
          RandomForestClassifier(),
          GradientBoostingClassifier(),
          DecisionTreeClassifier(),
          RidgeClassifier(),
          SVC()
          ]

for m in models:
    print(f'Current Model is {m}')

    m.fit(X_train, Y_train)

    pred_train = m.predict(X_train)
    print(f'Train Accuracy is : {acs(Y_train, pred_train)}')

    pred_test = m.predict(X_test)
    print(f'Test Accuracy is : {acs(Y_test, pred_test)}\n')