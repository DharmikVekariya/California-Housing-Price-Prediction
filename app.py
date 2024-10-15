import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# Load the saved model and scaler
@st.cache_resource
def load_model_and_scaler():
    df = pd.read_csv('housing.csv')
    df.dropna(inplace=True)

    X = df.drop(['median_income'],axis=1)
    y = df['median_income']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
   
    train_df = X_train.join(y_train)
    train_df['total_rooms'] = np.log(train_df['total_rooms']+1)
    train_df['total_bedrooms'] = np.log(train_df['total_bedrooms']+1)
    train_df['population'] = np.log(train_df['population']+1)
    train_df['households'] = np.log(train_df['households']+1)
    train_df = pd.get_dummies(train_df)
    train_df['bedroom_ratio'] = train_df['total_bedrooms']/train_df['total_rooms']
    train_df['households_rooms'] = train_df['total_rooms']/train_df['households']
    scaler = StandardScaler()
    X_train,y_train=train_df.drop(['median_house_value'],axis=1),train_df['median_house_value']
    X_train_s = scaler.fit_transform(X_train)
    forest = RandomForestRegressor()
    forest.fit(X_train_s,y_train)

    return forest,scaler

model, scaler = load_model_and_scaler()

# Title and description
st.title("California Housing Price Prediction")
st.write("""
This app predicts housing prices in California using a pre-trained machine learning model.
Enter the required features, and the model will predict the median house value.
""")

# Feature inputs
st.subheader("Input Features")

# Numeric input fields for numeric features
longitude = st.number_input("Longitude", value=-120.0)
latitude = st.number_input("Latitude", value=35.0)
housing_median_age = st.number_input("Housing Median Age", min_value=0, max_value=100, value=25)

# Applying log transformations to user inputs
total_rooms = st.number_input("Total Rooms", min_value=0, max_value=50000, value=3000)
total_bedrooms = st.number_input("Total Bedrooms", min_value=0, max_value=10000, value=500)
population = st.number_input("Population", min_value=0, max_value=50000, value=1500)
households = st.number_input("Households", min_value=0, max_value=10000, value=500)

# Median income
median_income = st.number_input("Median Income (in tens of thousands)", value=5.0)

# Ratio features
bedroom_ratio = total_bedrooms/total_rooms
households_rooms = total_rooms/households

# Categorical input for 'ocean_proximity'
st.subheader("Select Ocean Proximity Category")
ocean_proximity = st.selectbox(
    'Ocean Proximity',
    ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
)

# One-hot encode the 'ocean_proximity' input
ocean_proximity_features = {
    'ocean_proximity_<1H OCEAN': 0,
    'ocean_proximity_INLAND': 0,
    'ocean_proximity_ISLAND': 0,
    'ocean_proximity_NEAR BAY': 0,
    'ocean_proximity_NEAR OCEAN': 0
}

if ocean_proximity == '<1H OCEAN':
    ocean_proximity_features['ocean_proximity_<1H OCEAN'] = 1
elif ocean_proximity == 'INLAND':
    ocean_proximity_features['ocean_proximity_INLAND'] = 1
elif ocean_proximity == 'ISLAND':
    ocean_proximity_features['ocean_proximity_ISLAND'] = 1
elif ocean_proximity == 'NEAR BAY':
    ocean_proximity_features['ocean_proximity_NEAR BAY'] = 1
elif ocean_proximity == 'NEAR OCEAN':
    ocean_proximity_features['ocean_proximity_NEAR OCEAN'] = 1

# Apply log transformations to the selected features
log_total_rooms = np.log(total_rooms + 1)
log_total_bedrooms = np.log(total_bedrooms + 1)
log_population = np.log(population + 1)
log_households = np.log(households + 1)

# Prepare the input data in the correct order
input_data = np.array([[
    longitude,                  # 'longitude'
    latitude,                   # 'latitude'
    housing_median_age,          # 'housing_median_age'
    log_total_rooms,             # 'total_rooms' (log-transformed)
    log_total_bedrooms,          # 'total_bedrooms' (log-transformed)
    log_population,              # 'population' (log-transformed)
    log_households,              # 'households' (log-transformed)
    median_income,               # 'median_income'
    ocean_proximity_features['ocean_proximity_<1H OCEAN'],  # 'ocean_proximity_<1H OCEAN'
    ocean_proximity_features['ocean_proximity_INLAND'],     # 'ocean_proximity_INLAND'
    ocean_proximity_features['ocean_proximity_ISLAND'],     # 'ocean_proximity_ISLAND'
    ocean_proximity_features['ocean_proximity_NEAR BAY'],   # 'ocean_proximity_NEAR BAY'
    ocean_proximity_features['ocean_proximity_NEAR OCEAN'], # 'ocean_proximity_NEAR OCEAN'
    bedroom_ratio,              # 'bedroom_ratio'
    households_rooms            # 'households_rooms'
]])

# Apply the StandardScaler to the input data
scaled_input_data = scaler.transform(input_data)

# Make predictions
if st.button("Predict Housing Price"):
    prediction = model.predict(scaled_input_data)
    st.write(f"Predicted Median House Value: ${prediction[0]:,.2f}")
