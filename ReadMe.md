# Flood Prediction WebApp

A web application built using **Streamlit** that predicts the likelihood of floods based on weather data. The app utilizes a machine learning model that is a hybrid of **LSTM (Long Short-Term Memory)** and **RNN (Recurrent Neural Network)** architectures, achieving an accuracy of **86%**. The app fetches real-time weather data using the **OpenWeather API** and provides flood prediction outputs based on this data.

## Features

- **Flood Prediction**: Predicts the likelihood of floods based on weather conditions (such as rainfall, temperature, and humidity).
- **Real-time Data**: Fetches live weather data using the OpenWeather API.
- **User-friendly Interface**: Interactive and easy-to-use interface created with Streamlit.
- **Accuracy**: The model achieves an **86% accuracy** in predicting floods.

## Tech Stack

- **Machine Learning**: 
  - **LSTM (Long Short-Term Memory)** and **RNN (Recurrent Neural Network)** hybrid model for flood prediction.
  - **TensorFlow** / **Keras** for model development and training.
- **Web Framework**: 
  - **Streamlit** for building the interactive web application.
- **API Integration**: 
  - **OpenWeather API** to fetch real-time weather data.
- **Programming Languages**: 
  - **Python** for backend and model implementation.
  - **JavaScript** (optional if used for front-end interactivity).
- **Data Storage**: Data is fetched dynamically from OpenWeather API, no persistent data storage used in the app.

## Accuracy

- **Model Type**: Hybrid LSTM + RNN.
- **Model Accuracy**: **86%** on the validation dataset.
- **Performance**: The model has been trained to predict the likelihood of floods based on historical weather data and current weather conditions, achieving high prediction accuracy.

## Screenshot
![1](https://github.com/user-attachments/assets/d942132b-b93c-4156-a9b0-c34b9dd35e2d)


