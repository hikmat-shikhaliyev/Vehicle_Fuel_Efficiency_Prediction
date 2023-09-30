# Vehicle Fuel Efficiency Prediction
This repository contains Python code for predicting the fuel efficiency (miles per gallon) of vehicles based on their specifications. The project utilizes regression models to make these predictions.

# Project Structure
The project is structured as follows:

Data: The project uses a dataset located at C:\Users\ASUS\Downloads\auto-mpg.csv. This dataset contains various specifications of vehicles such as displacement, horsepower, weight, acceleration, model year, origin, and miles per gallon (mpg).

Code: The main code file is named vehicle_fuel_efficiency_prediction.py. This file contains Python code to read the dataset, preprocess the data, train regression models (Random Forest Regressor), optimize the model, and evaluate the model's performance.

# Data Preprocessing
The initial dataset is loaded using pandas and basic exploratory data analysis is performed.
Irrelevant columns such as 'car name' are dropped.
Missing values are handled appropriately.
Categorical variables are reduced to numerical using label encoding.
Outliers are detected and treated using the IQR method.

# Model Training and Optimization
The dataset is split into training and testing sets.
A Random Forest Regressor model is trained on the data.
The Random Forest Regressor is further optimized using Randomized Search Cross-Validation.
Feature importance is assessed and less relevant features are removed for model optimization.

# Model Evaluation
Model evaluation metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) score.
R-squared score is calculated to measure the goodness of fit of the model.

# Result Interpretation
The script will output the evaluation metrics (MAE, MSE, RMSE, R2) for both the training and testing sets. These metrics provide insights into the accuracy and goodness of fit of the regression model.
