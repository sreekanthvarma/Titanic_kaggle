# Loading required libraries
import numpy as np
import pandas as pd

# loading datasets
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

x = len(train)

# separating out target variable 
target = train["Survived"]
del train["Survived"]

# joining train and test data set to clean and modify
data = pd.concat([train,test])

# looking for percentage of missing values 
data.isnull().sum()/len(data)

# selecting features which are more likely to influence
to_train = data[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]]

to_train.dtypes

# converting two features to int format and later to categorical variables
to_train["Embarked"].unique()
to_train["Sex"].unique()

convert = {"Embarked":  {"S":0,"C":1,"Q":2},
           "Sex":       {"male":1, "female":0}}

to_train.replace(convert, inplace=True)

# imputing missing values by filling with median value
median_age = to_train["Age"].median()
to_train["Age"].fillna(median_age, inplace = True)

## creating a new feature based on age and sex
# all women are classified as 0 as they were given priority as per the data
# children under 15 years had more probability of surviving, so they were given 1
# other categories were created using similar criteria of chances of survival
agesex = []
    
for age in to_train['Age']:
    if age < 15:
        agesex.append(1)
    elif age>=15 and age<20:
        agesex.append(2)
    elif age>=20 and age<50:
        agesex.append(3)
    else:
        agesex.append(4)

to_train['AgeSex'] = agesex        

to_train['AgeSex'] = np.where(to_train['Sex'] == 0, 0,to_train['AgeSex'])
    
to_train.head(20)

# to categorical variables
to_train["Embarked"] = to_train["Embarked"].astype('category')
to_train["Sex"] = to_train["Sex"].astype('category')
to_train["AgeSex"] = to_train["AgeSex"].astype('category')


to_train.head()
to_train.dtypes


median_fare = to_train["Fare"].median()
to_train["Fare"].fillna(median_fare, inplace = True)

# since majority are in 0 category, replaced the 2 missing values with 0
to_train["Embarked"].fillna(0, inplace = True)

to_train.isnull().sum()

# standardizing by subtracting mean and dividing by standard deviation
to_train['Fare'] = (to_train['Fare'] - to_train['Fare'].mean())/to_train['Fare'].std()

to_train['Age'] = (to_train['Age'] - to_train['Age'].mean())/to_train['Age'].std()

to_train.head()

# Separate the data back to train and test
train_data = to_train.iloc[:x,]
test_data = to_train.iloc[x:, ]

len(train_data)
train_data.head()

## Prediction
##-----------------------------------------------------------------
# Loading Logistic Regression model to predict the survival rate
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(train_data, target)

prediction = pd.DataFrame(logreg.predict(test_data))

index = pd.DataFrame(list(range(892, 1310)))


output = pd.concat([index, prediction], axis = 1)
output.columns = ['PassengerId','Survived']
output.set_index('PassengerId', inplace=True)

output.to_csv("result/titanic_2.csv")

##---------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
rdf = RandomForestClassifier(n_estimators = 100)

rdf.fit(train_data, target)

prediction_rdf = rdf.predict(test_data)

output_rdf = pd.concat([index, prediction], axis = 1)
output_rdf.columns = ['PassengerId','Survived']

output_rdf.set_index('PassengerId', inplace=True)

output_rdf.to_csv("result/titanic_2_rdf.csv")
