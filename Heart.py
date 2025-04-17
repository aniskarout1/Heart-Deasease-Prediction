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

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=40)

age=input("Enter your age: ")
sex=input("Enter your gender 0 for female, 1 for male: ")
cp=input("Enter your chaist pain type 0: Typical angina (chest pain due to reduced blood flow) 1: Atypical angina (chest pain not related to heart disease) 2: Non-anginal pain (other causes of chest pain) 3: Asymptomatic (no chest pain): ")
trestbps=input("Enter your Blood pressure level: ")
chol=input("Enter your cholestor level: ")
fbs=input("Enter your fasting Blood sugar: ")
restecg=input("Enter your Electrocardiographic results 0: Normal 1: ST-T wave abnormality 2: Left ventricular hypertrophy (LVH): ")
thalach=input("Enter your Maximum heart rate achieved: ")
exang=input("Enter your exercise-induced angina, Whether the patient experienced angina during exercise (1 = yes, 0 = no): ")
oldpeak=input("Enter your oldpeak, Depression in the ST segment during exercise: ")
slope=input("Enter your Slope of the ST segment during peak exercise 0: Upsloping (normal) 1: Flat (potential heart disease) 2: Downsloping (higher risk of heart disease): ")
ca=input("Enter your number of major blood vessels (0–3) visible in fluoroscopy: ")
thal=input("Enter your thalassemia test result 1: Normal2: Fixed defect (permanent damage)3: Reversible defect (potential reversible damage): ")

input_data=(float(age),int(sex),int(cp),float(trestbps),float(chol),int(fbs),int(restecg),float(thalach),int(exang),float(oldpeak),int(slope),int(ca),int(thal))
arr=np.array(input_data)
prediction=model.predict(arr.reshape(1,-1))
print(prediction)
if(prediction==0):
    print("You are healthy")
else:
    print("You should consult your doctor")