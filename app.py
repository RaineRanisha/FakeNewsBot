import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("RayOfLife/distilbert-fake-news")
    model = DistilBertForSequenceClassification.from_pretrained("RayOfLife/distilbert-fake-news")
    return model, tokenizer

model, tokenizer = load_model()
model.eval()

st.title("Fake News Detection Chatbot")
st.write("Enter a news article or headline, and this chatbot will predict whether it is likely Fake or True.")

text = st.text_area("Enter news text below:", height=200)

if st.button("Check"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).cpu().numpy()[0]
            confidence = probs[0][predicted_class].item()

        label = "True News" if predicted_class == 1 else "Fake News"
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
