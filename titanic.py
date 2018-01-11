import numpy as np
import pandas as pd

# loading datasets
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

target = train["Survived"]
x = len(train)

del train["Survived"]

data = pd.concat([train,test])
len(data)
data.head()

data.isnull().sum()/len(data)

to_train = data[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]]

to_train.dtypes

to_train["Embarked"].unique()
to_train["Sex"].unique()

convert = {"Embarked":  {"S":0,"C":1,"Q":2},
           "Sex":       {"male":1, "female":0}}

to_train.replace(convert, inplace=True)

to_train["Embarked"] = to_train["Embarked"].astype('category')
to_train["Sex"] = to_train["Sex"].astype('category')

to_train.head()
to_train.dtypes

#from sklearn.preprocessing import Imputer

#imp = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)

#imp.fit_transform(to_train)

median_age = to_train["Age"].median()
to_train["Age"].fillna(median_age, inplace = True)

median_fare = to_train["Fare"].median()
to_train["Fare"].fillna(median_fare, inplace = True)

to_train["Embarked"].fillna(0, inplace = True)

to_train.isnull().sum()

to_train['Fare'] = (to_train['Fare'] - to_train['Fare'].mean())/to_train['Fare'].std()

to_train['Age'] = (to_train['Age'] - to_train['Age'].mean())/to_train['Age'].std()

to_train.head()
# Separate the data back to train and test
train_data = to_train.iloc[:x,]
test_data = to_train.iloc[x:, ]

len(train_data)
train_data.head()
target.head()

# Loading Logistic Regression model to predict the survival rate
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(train_data, target)

prediction = pd.DataFrame(logreg.predict(test_data))

index = pd.DataFrame(list(range(892, 1310)))


output = pd.concat([index, prediction], axis = 1)
output.columns = ['PassengerId','Survived']
output.set_index('PassengerId', inplace=True)

output.to_csv("result/titanic.csv")

##---------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
rdf = RandomForestClassifier(n_estimators = 100)

rdf.fit(train_data, target)

prediction_rdf = rdf.predict(test_data)

output_rdf = pd.concat([index, prediction], axis = 1)
output_rdf.columns = ['PassengerId','Survived']

output_rdf.set_index('PassengerId', inplace=True)

output_rdf.to_csv("result/titanic_rdf.csv")

