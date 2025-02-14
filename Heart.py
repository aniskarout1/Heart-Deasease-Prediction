import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv("heart_disease_data.csv")
print(df.head())
# print(df.tail())
# print(df.describe())
# print(df["target"].value_counts())
x=df.drop(columns="target", axis=1)
# print(x)
y=df["target"]
# print(y)

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
# print(x_train.shape,x_test.shape)

model=LogisticRegression()
model.fit(x_train,y_train)
predict=model.predict(x_test)
# print(predict)
# print(accuracy_score(y_test,predict))

age=input("Enter your age: ")
sex=input("Enter your sex: ")
cp=input("Enter your cp: ")
trestbps=input("Enter your trestbps: ")
chol=input("Enter your cholestor level: ")
fbs=input("Enter your fbs: ")
restecg=input("Enter your restecg: ")
thalach=input("Enter your thalach: ")
exang=input("Enter your exang: ")
oldpeak=input("Enter your oldpeak: ")
slope=input("Enter your slope: ")
ca=input("Enter your ca: ")
thal=input("Enter your thal: ")

input_data=(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
arr=np.array(input_data)
prediction=model.predict(arr.reshape(1,-1))
print(prediction)
if(prediction==0):
    print("You are healthy")
else:
    print("You should consult your doctor")