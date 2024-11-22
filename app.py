import streamlit as st
import requests
import numpy as np
import pickle
import tensorflow as tf

# Load the pre-trained model and scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model = tf.keras.models.load_model(r'C:\Users\ASUS\Desktop\FInal v2\hybrid_model (1).h5')

# Function to fetch real-time weather data from OpenWeatherMap API
def fetch_weather_data(city):
    api_key = '8a410f803a8a68420e588e1e0322ce99'  # Replace with your API key
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(base_url)
    data = response.json()
    
    if response.status_code == 200:
        main = data['main']
        wind = data['wind']
        weather_data = {
            'temperature': main['temp'],
            'humidity': main['humidity'],
            'pressure': main['pressure'],
            'wind_speed': wind['speed'],
            'rainfall': data['rain']['1h'] if 'rain' in data else 0
        }
        return weather_data
    else:
        st.error(f"Error fetching data for {city}. Please check the city name.")
        return None

# Function to predict flood risk using real-time weather data
def predict_flood_risk(features):
    scaled_features = scaler.transform(np.array(features).reshape(1, -1))
    scaled_features = scaled_features.reshape(scaled_features.shape[0], scaled_features.shape[1], 1)
    prediction = model.predict(scaled_features)
    return prediction

# Predefined location data
location_data = {
    'Pune': [1,5,4,6,8,2,7,5,6,9,9,2,1,4,7,2,4,8,2,6],
    'buldana':[9,	9	,9	,8	,4	,5	,3	,6	,4	,4	,5	,7	,12,	3,	3,	4,	12,	9,	3,	11],
    'chandrapur':[9,	9	,9	,8	,4	,5	,3	,6	,4	,4	,5	,7	,0,	3,	3,	4,	2,	9,	3,	1],
    'akola':[2,	7	,9	,8	,4	,5	,3	,6	,4	,4	,5	,7	,5,	3,	3,	0,	12,	9,	3,	1],
    'amravati':[1,	8	,8	,8	,4	,5	,3	,6	,4	,4	,5	,7	,4,	3,	3,	4,	10,	1,	3,	1],
    'Nagpur': [1,5,4,6,8,2,7,5,6,9,4,2,1,4,9,2,4,8,2,6],
    'Mumbai': [2,	7	,9	,8	,4	,5	,3	,6	,4	,4	,0	,7	,5,	3,	3,	4,	6,	9,	3,	1]
}

# Streamlit app layout
st.title("Flood Prediction using LSTM and RNN")

# Select mode for input: Real-time weather data or predefined data
input_mode = st.radio("Select Input Mode:", ("Predefined Location Data", "Real-time Weather Data"))

if input_mode == "Predefined Location Data":
    # Select predefined location
    location = st.selectbox("Select a predefined location:", list(location_data.keys()))

    if st.button('Predict using Predefined Data'):
        features = location_data[location]
        result = predict_flood_risk(features)
        
        # st.write(f"Input features for {location}: {features}")
        if result > 0.5:
            st.write(f"There is a high probability of a flood in {location}. Risk: {result[0][0]*100:.2f}%")
        else:
            st.write(f"There is a low probability of a flood in {location}. Risk: {result[0][0]*100:.2f}%")

elif input_mode == "Real-time Weather Data":
    # Input for real-time location
    location = st.text_input("Enter a location (e.g., Patna, Bihar):", "Patna")

    if st.button('Get Prediction with Real-time Data'):
        if location:
            weather_data = fetch_weather_data(location)

            if weather_data:
                st.write(f"Weather Data for {location}:")
                st.write(f"Temperature: {weather_data['temperature']} °C")
                st.write(f"Humidity: {weather_data['humidity']} %")
                st.write(f"Pressure: {weather_data['pressure']} hPa")
                st.write(f"Wind Speed: {weather_data['wind_speed']} m/s")
                st.write(f"Rainfall: {weather_data['rainfall']} mm")

                # Assuming your model expects 20 features, the first five are fetched from OpenWeatherMap
                input_features = [
                    weather_data['temperature'], weather_data['humidity'], weather_data['pressure'],
                    weather_data['wind_speed'], weather_data['rainfall'],  # Actual weather data
                    1,7,6,2,1,1,1,4,3,6,1,5,2,4,2 # Placeholder values for other inputs
                ]

                prediction = predict_flood_risk(input_features)

                if prediction > 0.5:
                    st.write(f"There is a high probability of a flood in {location}. Risk: {prediction[0][0]*100:.2f}%")
                else:
                    st.write(f"There is a low probability of a flood in {location}. Risk: {prediction[0][0]*100:.2f}%")
        else:
            st.error("Please enter a valid location.")
