import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load your trained sentiment analysis model
model = tf.keras.models.load_model('sentiment_model.h5')


# Streamlit app title and description
st.title("Tweet Sentiment Analysis")
st.write("Enter a tweet below to predict its sentiment.")

# Input text box
input_text = st.text_area("Enter tweet:")

# Function to preprocess the input text
def preprocess_input(text):
    # Tokenize the input text
    input_sequence = Tokenizer.texts_to_sequences([text])  # Convert to sequences
    # Pad the sequence to match the input length expected by the model
    input_data = pad_sequences(input_sequence, maxlen=100)  # Set maxlen as per your model's requirement
    return input_data

# Add a submit button
submit_button = st.button("Submit")

if submit_button and input_text:
    input_data = preprocess_input(input_text)

    # Make prediction
    prediction = model.predict(input_data)

    # If it's a binary classification (single output), perform the check
    if prediction.shape[1] == 1:  # Binary classification
        prediction_value = prediction[0][0]
        sentiment = "Positive" if prediction_value > 0.5 else "Negative"
        st.write(f"Prediction: {sentiment} (Confidence: {prediction_value:.2f})")

    # If it's a multi-class classification, use argmax to get the index of the highest probability
    else:
        predicted_class = prediction.argmax(axis=-1)  # Gets the index of the highest value
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        sentiment = sentiment_labels[predicted_class[0]]
        st.write(f"Prediction: {sentiment} (Confidence: {prediction[0][predicted_class[0]]:.2f})")
