import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"C:\Users\HP-PC\Desktop\Python data analytics\sentiment analysis project\sentiment_model.h5")



@st.cache_resource
def load_tokenizer():
    with open(r"C:\Users\HP-PC\Desktop\Python data analytics\sentiment analysis project\tokenizer.pickle", "rb") as handle:
        return pickle.load(handle)



def preprocess_text(text, tokenizer, max_length=100):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded


st.title("Sentiment Analysis with RNN")
st.write("Enter a sentence to analyze its sentiment.")


user_input = st.text_area("Input Text:")


model = load_model()
tokenizer = load_tokenizer()

if st.button("Analyze Sentiment"):
    if user_input:
        processed_text = preprocess_text(user_input, tokenizer)
        prediction = model.predict(processed_text)[0][0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        st.write(f"Predicted Sentiment: **{sentiment}** (Score: {prediction:.2f})")
    else:
        st.warning("Please enter some text to analyze.")