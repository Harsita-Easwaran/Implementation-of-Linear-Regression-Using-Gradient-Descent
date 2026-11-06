# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and preprocess the dataset by importing the required libraries, reading the CSV file, applying one-hot encoding to the categorical State column, and standardizing the input features and target values.

2.Prepare the data for training by separating the independent features (X) and target variable (y), then adding a bias (intercept) column to the feature matrix.

3.Train the Linear Regression model using Batch Gradient Descent by initializing the parameters, repeatedly updating them to minimize the Mean Squared Error (MSE) cost function, and plotting the cost convergence graph.

4.Predict the profit for a new startup by preprocessing the new input in the same way as the training data, applying the learned parameters to make a prediction, and converting the predicted value back to the original scale.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: HARSITA EASWARAN
RegisterNumber:  212224220036
*/
```
```
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
def linear_regression (X1,y, learning_rate=0.01, num_iters=1000) :
    X=np.c_[np. ones (len(X1)), X1]
    theta=np.zeros (X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y) .reshape(-1,1)
        theta=learning_rate*(1/len(X1))*X.T.dot (errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
X=(data.iloc[1:, :-2].values)
X1=X. astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler. fit_transform (X1)
Y1_Scaled=scaler. fit_transform(y)
print(data. head ()
     )
theta=linear_regression (X1_Scaled, Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler. fit_transform (new_data)
prediction=np.dot(np. append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler. inverse_transform (prediction)
print(f"Predicted value: {pre}")
```

## Output:
<img width="1180" height="276" alt="image" src="https://github.com/user-attachments/assets/3e3983a8-46a4-4658-86b2-38294ec5f268" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
