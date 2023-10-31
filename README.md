# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas to read the CSV file.
2. Load and check for null or duplicated values in the dataset.
3. Use LabelEncoder to encode the dataset values.
4. Import LogisticRegression and apply it to the dataset with train and test data (X and y).
5. Predict values using the variable y_pred.
6. Calculate accuracy, confusion, and classification report using sklearn's metrics.
7. Apply new data and print accuracy, confusion, and classification report.
8. End the program.

## Program:
```


Developed by: Vikash s
RegisterNumber: 212221040115

```
```
import pandas as pd

data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()

data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x

x = data1.iloc[:,:-1]
x
y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
## HEAD OF THE DATA :
![image](https://github.com/divz2711/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121245222/78837c62-fd5b-46e2-bf87-c25b394cba24)

## COPY HEAD OF THE DATA:
![image](https://github.com/divz2711/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121245222/9ece7191-2f84-4d68-a266-6e68355da220)

## NULL AND SUM :
![image](https://github.com/divz2711/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121245222/76732966-509e-4534-8990-6118895da69d)

## DUPLICATED :
![image](https://github.com/divz2711/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121245222/f81e060c-b131-4508-86f6-1747b2e99eaf)

## X VALUE:
![image](https://github.com/divz2711/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121245222/4298fcf2-8a52-4f96-a251-5e598e97a166)

## Y VALUE :
![image](https://github.com/divz2711/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121245222/5a53aa8d-f530-4e33-a349-d57c3e8aa3e3)

## PREDICTED VALUES :
![image](https://github.com/divz2711/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121245222/cd5403a0-f517-4e4e-97f6-62fc7cc0ba15)

## ACCURACY :
![image](https://github.com/divz2711/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121245222/ebed675f-5854-4dd5-9623-f7b7da87aae7)

## CONFUSION MATRIX :
![image](https://github.com/divz2711/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121245222/ca7684dc-e5e0-4054-a06b-16098c3cd37c)
## CLASSIFICATION REPORT :
![image](https://github.com/divz2711/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121245222/03b56914-984f-4d45-a81b-05867bcc5b0a)

## Prediction of LR :
![image](https://github.com/divz2711/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121245222/a12fb720-eed0-4830-812e-02b1a1e15acd)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.












