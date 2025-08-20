import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "alandh/sentimen-indobert"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Label mapping (adjust according to your training)
label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

def clean_text(text):
    return text.lower().strip()

def predict_sentiment(text):
    cleaned = clean_text(text)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        pred_label = torch.argmax(outputs.logits, dim=1).item()

    return label_mapping[pred_label]

st.set_page_config(page_title="Tokopedia Review Sentiment", page_icon="💬")
st.title("💬 Tokopedia Review Sentiment Classifier")

user_input = st.text_area("Enter a product review:", "")

if st.button("Predict Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter a review first.")