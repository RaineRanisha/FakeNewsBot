import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

st.title("Fake News Detection Chatbot")
st.write("Enter a news article or snippet to detect whether it is **Fake** or **True**.")

@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("RayOfLife/distilbert-fake-news")
    tokenizer = DistilBertTokenizerFast.from_pretrained("RayOfLife/distilbert-fake-news")
    model.to(torch.device("cpu"))  # Force CPU use
    return model, tokenizer

model, tokenizer = load_model()

user_input = st.text_area("Paste news article here:", height=200)

if st.button("Check"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(torch.device("cpu")) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            predicted_class = int(torch.argmax(probs, dim=1).cpu().item())
            confidence = float(probs[0][predicted_class].cpu().item())

        label = "True News" if predicted_class == 1 else "Fake News"
        st.markdown(f"### Prediction: {label}")
        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
