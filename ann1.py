# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 07:31:05 2020

@author: Shubhangi sakarkar
"""
"Data Preprocessing"

"Importing the libraries"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"importing the data"
df=pd.read_csv('Churn_Modelling.csv')
X=df.iloc[:,3:13]
y=df.Exited

"encoding the categorical features"
geography=pd.get_dummies(X['Geography'],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)


"concating to original dataset"
X=pd.concat([X,geography,gender],axis=1)

"dropping not required column"
X.drop(['Geography','Gender'],axis=1,inplace=True)

"creating training and testing sets"

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=22)


"sacling the features"
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


"Applying model"

import keras
from keras.models import Sequential
from keras.layers import Dense,LeakyReLU,PReLU,ELU,Dropout

"initilaizing ANN"
classifier=Sequential()

"creating input and 1 hidden layer"
classifier.add(Dense(units=6, input_dim=11,kernel_initializer='he_uniform',activation='relu'))

"adding second hidden layer"
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu'))

"adding output layer"
classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))


"compling ANN"
classifier.compile(optimizer='Adamax',loss='binary_crossentropy',metrics=['accuracy'])

"fitting ANN to training set"
model_history=classifier.fit(X_train,y_train,batch_size=10,validation_split=0.33,epochs=100)



# list all data in history

print(model_history.history.keys())
# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
