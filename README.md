## EX 04-Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student


## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:


1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Vikash s
RegisterNumber:  212222240115

#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the file
dataset = pd.read_csv('Placement_Data_Full_Class.csv')
dataset

dataset.head(20)

dataset.tail(20)

#droping tha serial no salary col
dataset = dataset.drop('sl_no',axis=1)
#dataset = dataset.drop('salary',axis=1)

dataset = dataset.drop('gender',axis=1)
dataset = dataset.drop('ssc_b',axis=1)
dataset = dataset.drop('hsc_b',axis=1)
dataset

dataset.shape

dataset.info()

#catgorising col for further labelling
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes

dataset.info()

dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset

dataset.info()

dataset

#selecting the features and labels
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2)
dataset.head()

x_train.shape

x_test.shape

y_train.shape

y_test.shape

from sklearn.linear_model import LogisticRegression
clf= LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)

clf.predict([[0, 87, 0, 95, 0, 2, 78, 2, 0]])
*/
```

## Output:

DATASET:

![1](https://github.com/Pavithraramasaamy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118596964/bb363891-c5a6-4152-b013-f92e1b471268)


dataset.head():

![2](https://github.com/Pavithraramasaamy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118596964/2f9a49ed-b0c5-4ddf-976f-ecaba21a4258)

dataset.tail():

![3](https://github.com/Pavithraramasaamy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118596964/5fbb88a6-e1f4-4c8b-8d7c-76dd56acda69)

dataset after dropping:

![4](https://github.com/Pavithraramasaamy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118596964/68d41dd6-302d-4150-8d72-2f4b74679145)



![5](https://github.com/Pavithraramasaamy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118596964/ff82ece4-fadf-4719-a44f-c82dddf0d8ba)

datase.shape:

![6](https://github.com/Pavithraramasaamy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118596964/df512678-e13b-4be6-b1cf-d77f4a549cdf)

dataset.info()

![7](https://github.com/Pavithraramasaamy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118596964/16088223-bfad-4ff8-85f6-e360871d141a)

dataset.dtypes

![8](https://github.com/Pavithraramasaamy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118596964/d1e04c04-4885-4b76-8052-c01033440daf)

dataset.info()

![9](https://github.com/Pavithraramasaamy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118596964/74e74692-27bb-47a8-9262-26f2246e1532)

dataset.codes

![10](https://github.com/Pavithraramasaamy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118596964/63d2b419-7ca1-4fec-97b1-0e12c70409c8)

selecting the features and labels


![13](https://github.com/Pavithraramasaamy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118596964/dfc5965b-008c-4a1d-b438-f09f49dcb418)

dataset.head()

![14](https://github.com/Pavithraramasaamy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118596964/580b9de7-1a34-4f86-b04d-25ad3e0947d8)

x_train.shape


![15](https://github.com/Pavithraramasaamy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118596964/d3f3e3f5-c7f6-4e5f-9da9-d5b7bce83af9)

x_test.shape

![16](https://github.com/Pavithraramasaamy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118596964/bf515290-c19e-4bf8-8578-d819977d0fc4)

y_train.shape

![17](https://github.com/Pavithraramasaamy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118596964/5f370c89-fa25-4621-af89-ffedce3d1edc)

y_test.shape


![18](https://github.com/Pavithraramasaamy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118596964/5e50d98e-8971-42be-b274-1d5e7bf95daf)


![21](https://github.com/Pavithraramasaamy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118596964/eccb4d00-a5ba-4dc1-a5cf-95995b767d4b)


clf.predict

![20](https://github.com/Pavithraramasaamy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118596964/f511211c-9e52-40b9-9a5b-5ea0700e523e)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

