# Real Estate Price Prediction Project

This project is a real estate price prediction system built using machine learning techniques. It predicts the price of a property based on features such as bedrooms, bathrooms, area, and location.

## Project Overview

The goal of this project is to create a simple web application that allows users to input property details and get an estimated price. The project consists of the following components:

- `housingProject.py`: Python script containing data preprocessing, data analysis, model training, and prediction functions.
- `data.csv`: CSV file containing the dataset used for training and testing the model. **(the data set is obtained from Kaggle: https://www.kaggle.com/datasets/danishjmeo/karachi-housing-prices-2023?datasetId=3346325)**
- `app.py`: Flask web application serving as the backend of the project.
- `model.pickle`: Pickle file containing the trained machine learning model.
- `columns.json`: JSON file containing the column names used during training.
- `index.html`: Containing a simple frontend.

## Model Results
Forest Tree Regression was used in this project which yielded the following values of MSE RMSE and R-SQUARED:
![error values](/Screenshots/value_of_errors.png)
<br> <br> Scatter Plot of the actual and predicted values after testing the model:
![chart](/Screenshots/chart.png)

## Front-end Demonstration
following is a short gif of the web application. (the value of the predicted price is changing as the value of Bedrooms, Bathrooms, Area, Location, and type is changed and the button is pressed).
![video](/Screenshots/webapp_demonstration.gif)

## Running the app
- Download the repository
- Run the app.py file
- A URL will be assigned for e.g. (http://127.0.0.1:5000)
- copy and paste it on any browser and run it and you're good to go!

## Requirements

To run this project, you need to download the following dependencies (you can download them using cmd etc):
numpy, pandas, scikit-learn, scipy, appdirs, certifi, click, distlib, Flask, Flask-Cors, gunicorn, itsdangerous, Jinja2, joblib, MarkupSafe, python-dateutil, pytz, six, xgboost, virtualenv, Werkzeug, wincertstore.
