import streamlit as st
import tensorflow as tf
import re
import numpy as np

model = tf.keras.models.load_model('sentiment_model.keras')



# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', ' ', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Streamlit UI
st.title("Twitter Sentiment Analysis")
st.subheader("Enter a tweet below to analyze its sentiment")

# User input
user_input = st.text_area("Enter tweet:", "")


if st.button("Analyze Sentiment"):
    if user_input:
        processed_text = preprocess_text(user_input)
        
        # Convert text to the format expected by the model (modify if needed)
        input_data = np.array([processed_text])  # Adjust as per your model's input
        
        # Predict sentiment
        prediction = model.predict(input_data)
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        
        st.subheader(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter a tweet to analyze.")