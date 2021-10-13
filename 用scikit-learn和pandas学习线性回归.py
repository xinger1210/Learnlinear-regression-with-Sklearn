# -*- encoding = utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

data=pd.read_excel("Folds5x2_pp.xlsx")
# print(data.head())
# print(data.shape)
# print(data.describe())

x=data[["AT","V","AP","RH"]]
# print(x.head())
y=data[["PE"]]
# print(y.head())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.3,random_state=1)
# print (X_train.shape)
#
# print (y_train.shape)
# print (X_test.shape)
# print (y_test.shape)

from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X_train,y_train)
# print(linreg.score(X_train,y_train))
# print("***")
# print(linreg.coef_)
# print("***")
# print(linreg.intercept_)

y_pred=linreg.predict(X_test)

from sklearn import metrics

# print("mse:",metrics.mean_squared_error(y_test,y_pred))
# print("rmse:",np.sqrt(metrics.mean_squared_error(y_test,y_pred))
#       )


X = data[['AT', 'V', 'AP']]
y = data[['PE']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
#模型拟合测试集
y_pred = linreg.predict(X_test)
from sklearn import metrics

print ("mse:",metrics.mean_squared_error(y_test, y_pred))

print ("rmse:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("****")
#交叉验证

X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]
from sklearn.model_selection import cross_val_predict
predicted=cross_val_predict(linreg,X,y,cv=10)

print ("mse:",metrics.mean_squared_error(y, predicted))

print ("rmse:",np.sqrt(metrics.mean_squared_error(y, predicted)))
fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()