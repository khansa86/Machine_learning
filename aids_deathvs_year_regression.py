# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data=pd.read_csv('aids.csv')
X=data.iloc[:,1:-1].values
y=data.iloc[:,2:3].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'orange')
plt.plot(X_train, regressor.predict(X_train), color = 'blue') 
plt.title('Years vs Deaths (Training set)')
plt.xlabel('Year')
plt.ylabel('Deaths')
plt.show()

plt.scatter(X_test, y_test, color = 'red')#scatter plot of test data 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Years vs Deaths (Test set)')
plt.xlabel('Year')
plt.ylabel('Deaths')
plt.show()
print (regressor.predict([[1993]]))
