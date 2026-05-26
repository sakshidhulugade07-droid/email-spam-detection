import streamlit as tf
import streamlit as st
import os
import pickle

# Set page configuration
st.set_page_config(page_title="Spam Classifier", page_icon="✉️", layout="centered")

# Title and description
st.title("✉️ AI Spam Email Detector")
st.markdown("Enter the content of an email or message below to verify if it's safe (**Ham**) or suspicious (**Spam**).")

# Function to load the ML model safely
@st.cache_resource
def load_model():
    model_path = 'spam_detector_model.pkl'
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as file:
        return pickle.load(file)

model = load_model()

# Check if model exists; if not, throw a warning on the UI
if model is None:
    st.error("⚠️ **Model file not found!** Please run `python train.py` in your terminal first to train and generate the model.")
else:
    # Text input container
    user_input = st.text_area("Paste Email/Message Content Here:", height=150, placeholder="Type or paste your message text here...")
    
    if st.button("Analyze Message", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            # Run prediction
            prediction = model.predict([user_input])[0]
            probabilities = model.predict_proba([user_input])[0]
            
            # Extract confidence percentages
            # Class classes_ are alphabetically arranged: ['ham', 'spam']
            ham_prob = probabilities[0] * 100
            spam_prob = probabilities[1] * 100
            
            st.write("---")
            st.subheader("Analysis Result")
            
            # Display custom UI message status cards
            if prediction == 'spam':
                st.error(f"🚨 **Prediction: SPAM**")
                st.metric(label="Spam Confidence", value=f"{spam_prob:.2f}%")
                st.info("Advise: Avoid clicking any links or replying to this sender.")
            else:
                st.success(f"✅ **Prediction: HAM (Legitimate Email)**")
                st.metric(label="Safe/Ham Confidence", value=f"{ham_prob:.2f}%")
                st.info("Advise: This message looks completely normal.")