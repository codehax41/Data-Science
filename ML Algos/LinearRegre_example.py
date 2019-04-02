import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('Salary_Data.csv')
#preparing i/p, o/p variables
x=df['YearsExperience']
y=df['Salary']

print('x:\n', x)
print('\n y:\n', y)

#splitting the train, test data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

x_train = x_train.values.reshape(-1,1)
x_test = x_test.values.reshape(-1,1)

print('x_train:\n', x_train)
print('\nx_test:\n', x_test)

#Fit the simple linear regression model
from sklearn.linear_model import LinearRegression, SGDRegressor

regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set variables
y_pred = regressor.predict(x_test)
print('\ny_pred:\n', y_pred)
print('\ny_test:\n', y_test)
#
# #Visulaizing the training results
# plt.scatter(x_train,y_train,color='red')
# plt.plot(x_train,regressor.predict(x_train),color='blue')
# plt.xlabel('Years of exp')
# plt.ylabel('Salary')
# plt.title('Training data')
# plt.show()
#
# #Visualizing the test set
# plt.scatter(x_test,y_test,color='red')
# plt.plot(x_test,regressor.predict(x_test),color='blue')
# plt.xlabel('Years of exp')
# plt.ylabel('Salary')
# plt.title('Test data')
# plt.show()
#
# #Evolution matrix
# from sklearn import metrics
# print('\nMAE:',metrics.mean_absolute_error(y_true=y_test,y_pred=y_pred))
# print('\nMSE:',metrics.mean_squared_error(y_true=y_test,y_pred=y_pred))
# print('\nRMSE:',np.sqrt(metrics.mean_squared_error(y_true=y_test,y_pred=y_pred)))
#
#
# #R2
# r_square = metrics.r2_score(y_test,y_pred)
# print('\nr_square:', r_square)
#
import statsmodels.api as sm
model = sm.OLS(y_test, y_pred)
result = model.fit()

print('\nLinear.summary:', result.summary())

#Fit the SDG Regressor
sdg_regressor = SGDRegressor()
sdg_regressor.fit(x_train,y_train)

#predicting the test set variables
y_pred = sdg_regressor.predict(x_test)
print('\ny_pred:\n', y_pred)
print('\ny_test:\n', y_test)

#import statsmodels.api as sm
model1 = sm.OLS(y_test, y_pred)
result1 = model1.fit()

print('\nSDG .summary:', result1.summary())


