import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import numpy as np

# Load model and tokenizer from Hugging Face
model = DistilBertForSequenceClassification.from_pretrained("RayOfLife/distilbert-fake-news")
tokenizer = DistilBertTokenizerFast.from_pretrained("RayOfLife/distilbert-fake-news")

# Streamlit app UI
st.title("Fake News Detection Chatbot")
st.write("Enter a news article snippet to check if it's Fake or Real.")

user_input = st.text_area("Paste news article or headline here:", height=200)

if st.button("Check"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs).item()
            confidence = probs[0][predicted_class].item()

        label = "True News" if predicted_class == 1 else "Fake News"
        st.markdown(f"### Prediction: {label}")
        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
