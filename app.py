import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch.nn.functional as F

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("RayOfLife/distilbert-fake-news")
    model = DistilBertForSequenceClassification.from_pretrained("RayOfLife/distilbert-fake-news")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

st.title("Fake News Detection Chatbot")
st.write("Enter a news headline or article snippet:")

user_input = st.text_area("News Text", height=200)

if st.button("Check"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.detach().cpu()
            probs = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).numpy()[0]
            confidence = probs[0][predicted_class].item()

        result = "True News" if predicted_class == 1 else "Fake News"
        st.subheader(f"Prediction: {result}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
