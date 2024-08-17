# Module 2 Linear Regression "Waiting Times as the Park"

# after defining the table and collecting data, create a linear regression model

#linear_model is a class of the sklearn module if contain different functions for performing machine learning with linear models.
# The term linear model implies that the model is specified as a linear combination of features. Based on training data, the learning process computes one weight for each feature to form a model that can predict or estimate the target value

from sklearn import linear_model 

# create input and output from the table in SMW
# the first value in the data is the day
# the second value in the data is the TIME of day
# ORDER IS IMPORTANT

# day of week, time of day 

input_data = [ # input the data 
  [1, 9],
  [1, 10],
  [1, 11],
  [1, 12],
  [1, 13],
  [1, 14],
  [1, 15],
  [1, 16],
]

output_data = [0,10,20,30,40,30,20,10] # waiting time, out put the time to wait

# the next step provides machine learning with the data to the compiler
# This creates the linear model that we will be using and specify that we will be using the Linear Regression algorithm. There are many different types of models and algorithms used in artificial intelligence, so specifying both is required.
model = linear_model.LinearRegression()

# The fit is used to fill the input and output data above into the model that we have created. The input data is always first and then the output data is after.
model.fit(input_data, output_data)

# now its time to predict the outcome using the model
# this specific print statement is used to print the predicted wait time of the rides based on the data in the list
# [1, 11.5] predicts the wait time on Monday at 11:30
print("Waiting time is...")
print(model.predict([[1, 11.5]]))



#################################################################################
print() #space

from sklearn import linear_model #imports from sklearn library

# ORDER IS IMPORTANT

# day of week, price, number of customers waiting time
input_data = [
  [1, 5, 53, 9],
  [1, 7, 67, 10],
  [1, 10, 83, 11],
  [1, 4, 24, 12],
  [1, 14, 43, 13],
  [1, 6, 32, 14],
  [1, 3, 69, 15],
  [1, 8, 29, 16],
]
output_data = [0,10,20,30,40,30,20,10]

model = linear_model.LinearRegression()

model.fit(input_data, output_data)

# not very accurate as compared to polynominal regression
# time of day continues to grow instead of decreasing (the straight line in linear regression allows no curve to be seen))
print("Waiting time is...")
print(model.predict([ [1, 0, 1, 11.5] ]))
print(model.predict([ [1, 1, 3, 14] ]))
print(model.predict([ [2, 0, 2, 12] ]))
print("but this is not very accurate information.") 