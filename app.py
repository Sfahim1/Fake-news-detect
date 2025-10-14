import streamlit as st
import pickle




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

        # prediction == 1 -> Real, prediction == 0 -> Fake
        if prediction[0] == 1:
            st.success("The news is Real!")
        else:
            st.error("The news is Fake!")
    else:
        st.warning("Please enter some text to analyze.")