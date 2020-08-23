import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

#loading pre-existing data set set from sklearn
diabetes=datasets.load_diabetes()
#print(diadetes)
#printing key values of dataset i.e features of data
print(diabetes.keys())

# Selecting only one feature for implementing linearRegression i.e y= w0+w1*x

diabetes_X=diabetes.data[:,np.newaxis,2]

# Spliting the data into training and test data using list slicing
diabetes_x_train=diabetes_X[:-30]
diabetes_x_test=diabetes_X[-30:]

diabetes_y_train=diabetes.target[:-30] # target is a class used here to specify that these are the target values
diabetes_y_test=diabetes.target[-30:]

# form a linear model
model=linear_model.LinearRegression()
#fitting the model i.e training the model
model.fit(diabetes_x_train,diabetes_y_train)

#predicting the data i.e testing the data
diabetes_y_predict=model.predict(diabetes_x_test)

# Calculating mean sqruared error . MSE=(target_vaue-Predicted_value)
print("Mean Squared error is ",mean_squared_error(diabetes_y_test,diabetes_y_predict))

#Printing weights of the model. If single feature is selected then there will be only one weight.
print("weights ", model.coef_)

#ploting the graph for pictorial view
plt.scatter(diabetes_x_test,diabetes_y_test)
plt.plot(diabetes_x_test,diabetes_y_predict)

#intersecting point
print("Intercept ",model.intercept_)

plt.show()