import numpy as np
import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('diabetes.csv')

print(data.head())

# Get our Variables
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']

# Split the data

#split data
x_train, x_test, y_train, y_test= train_test_split(x,y)

# Scale our data

#scale data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

# Instantiate and train the model

#instatiate model
model = MLPClassifier(max_iter=1000, alpha=1)
model.fit(x_train, y_train)
print("Model Accuracy on training set:", model.score(x_train, y_train))
print("Model Accuracy on Test Set:", model.score(x_test, y_test))

# Create the function for Gradio

print(data.columns)
#create a function for gradio
def diabetes(Pregnancies, Glucose, Blood_Pressure, SkinThickness, Insulin,BMI,Diabetes_Pedigree, Age):
    x = np.array([Pregnancies,Glucose,Blood_Pressure,SkinThickness,Insulin,BMI,Diabetes_Pedigree,Age])
    prediction = model.predict(x.reshape(1, -1))
    return prediction


# Create our Gradio Interface

outputs = gr.Textbox()
app = gr.Interface(fn=diabetes,
inputs=['number','number','number','number','number','number','number','number'],
outputs=outputs,description="This is a diabetes model")

# Launch the Gradio Web App

#To provide a shareable link
app.launch(share=True)