import streamlit as st
import pickle

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.title("Fake News Detection")

user_input = st.text_area("Enter news text")

if st.button("Predict"):
    if user_input:
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)
        result = "Fake News" if prediction[0] == 1 else "Real News"
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some text.")