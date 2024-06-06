# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from itertools import combinations
%matplotlib inline
# Read the file "Advertising.csv"
df = pd.read_csv('Advertising.csv')
# Take a quick look at the data to list all the predictors
print(df.head())

# Initialize a list to store the MSE values
mse_list = []
Create different multi predictor models
# Create a list of lists of all unique predictor combinations
# For example, if you have 2 predictors,  A and B, you would 
# end up with [['A'],['B'],['A','B']]
predictors = [col for col in df.columns if col != 'Sales']
cols = []
for r in range(1, len(predictors) + 1):
    cols.extend([list(combo) for combo in combinations(predictors, r)])

# Loop over all the predictor combinations 
for i in cols:

    # Set each of the predictors from the previous list as x
    x =df[i]
    
    # Set the "Sales" column as the reponse variable
    y = df['Sales']
   
    # Split the data into train-test sets with 80% training data and 20% testing data. 
    # Set random_state as 0
    x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.2, random_state=0)

    # Initialize a Linear Regression model
    lreg = LinearRegression()

    # Fit the linear model on the train data
    lreg.fit(x_train, y_train)

    # Predict the response variable for the test set using the trained model
    y_pred = lreg.predict(x_test)
    
    # Compute the MSE for the test data
    MSE =  mean_squared_error(y_test, y_pred)
    
    # Append the computed MSE to the initialized list
    mse_list.append((i, MSE))
Display the MSE with predictor combinations
# Print the results using PrettyTable
from prettytable import PrettyTable
# Sort mse_list by MSE values
mse_list = sorted(mse_list, key=lambda x: x[1])

# Create a PrettyTable object
t = PrettyTable(['Predictors', 'MSE'])

# Loop through mse_list to add rows to the table
for predictors, mse in mse_list:
    t.add_row([predictors, round(mse, 3)])

# Print the table
print(t)
