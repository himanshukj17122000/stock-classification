#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:46:42 2019

@author: himanshu
"""
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
#importing the training set 
#only the np arrays can be the input to machine learning models in keras
dataset_train= pd.read_csv('Google_Stock_Price_Train 3.csv')
training_set= dataset_train.iloc[:,[1,5]].values

#standardising the data
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)

#Creating a data structure with 60 timestamps and 1 output 
x_train= []
y_train=[]

for i in range(60, 1258):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
x_train,y_train= np.array(x_train), np.array(y_train)

#reshaping 
# we can add new indicators using this line of code to add more dimension 
x_train= np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

#building the model 
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import LSTM
from keras.layers import Dropout 

#initialising the RNN 
regressor= Sequential()

#Adding the first LSTM layers and some Dropout regularisation 
regressor.add(LSTM(units=50, return_sequences= True, input_shape= (x_train.shape[1],1)))
regressor.add(Dropout(0.2))

#adding a second layer
regressor.add(LSTM(units=50, return_sequences= True))
regressor.add(Dropout(0.2))

#adding a third layer 
regressor.add(LSTM(units=50, return_sequences= True))
regressor.add(Dropout(0.2))

#adding the fourth layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

#adding the output layer 
regressor.add(Dense(units=1))

#compiling the RNN
regressor.compile(optimizer= 'adam', loss='mean_squared_error')

#Fitting the RNN to the training set
regressor.fit(x_train, y_train, epochs=10, batch_size=32)

#making the predictions and visualising the results 

#getting the real stock price of 2017 
dataset_test= pd.read_csv('Google_Stock_Price_Test 1.csv')
real_stock_price= dataset_test.iloc[:,[1,5]].values
real_stock_prices = dataset_test.iloc[:,1:2].values
#getting the predicted stock price of 2017 
dataset_total = pd.concat((dataset_train[['Open','Volume']], dataset_test[['Open','Volume']]), axis = 0)
inputs= dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,2)
inputs= sc.transform(inputs)
x_test= []

for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0])
x_test= np.asarray(x_test)
print(x_test)
x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
predicted_stock_price= regressor.predict(x_test)
predicted_stock_price2 = np.zeros(shape=(len(predicted_stock_price), 2) )
predicted_stock_price2[:,0]=predicted_stock_price[:,0]
predicted_stock_price= sc.inverse_transform(predicted_stock_price2)[:,0]

#visualizing the results 
plt.plot(real_stock_prices, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()




