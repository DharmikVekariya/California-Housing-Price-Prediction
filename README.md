# California Housing Price Prediction

This project aims to predict housing prices in California using the California Housing dataset. The model utilizes a Random Forest Regressor to achieve a prediction accuracy of 82.72%.
You can access the live demo of the application at the following link:

[California Housing Price Prediction App](https://california-housing-price-prediction-yutrbdatvjvhrfbfabjjuy.streamlit.app/)


## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Requirements](#Requirements)
- [Model Performance](#model-performance)
- [Contact](#Contact)

## Project Overview

The California Housing dataset contains information about various features of homes in California, such as location, median housing age, and income levels. The goal of this project is to build a machine learning model that can predict the median house value based on these features.

## Dataset

The dataset used in this project is sourced from the kaggle. The target variable is `median_house_value`.

## Features

The following features are used for predicting housing prices:

- `longitude`: Longitude of the location.
- `latitude`: Latitude of the location.
- `housing_median_age`: Median age of the houses.
- `total_rooms`: Total number of rooms in the area.
- `total_bedrooms`: Total number of bedrooms in the area.
- `population`: Population of the area.
- `households`: Number of households in the area.
- `median_income`: Median income of the area.
- `ocean_proximity_<1H OCEAN`: Categorical feature indicating proximity to the ocean.
- `ocean_proximity_INLAND`: Categorical feature indicating inland areas.
- `ocean_proximity_ISLAND`: Categorical feature indicating islands.
- `ocean_proximity_NEAR BAY`: Categorical feature indicating proximity to a bay.
- `ocean_proximity_NEAR OCEAN`: Categorical feature indicating proximity to the ocean.
- `bedroom_ratio`: Ratio of bedrooms to total rooms.
- `households_rooms`: Ratio of households to total rooms.

## Requirements

To run the project, you need the following dependencies:
- Python 3.x
- streamlit==1.39.0
- numpy==1.26.4
- pandas==2.0.3
- scikit-learn==1.5.2
- joblib==1.4.2

## Model Performance

The Random Forest Regressor achieved an accuracy of 81.5% on the test dataset. This performance indicates a reliable model for predicting housing prices in California based on the input features.

## Contact

If you have any questions or suggestions regarding the project, feel free to contact **Dharmik Vekariya** at **Vekariyadharmik2002@gmail.com**.
