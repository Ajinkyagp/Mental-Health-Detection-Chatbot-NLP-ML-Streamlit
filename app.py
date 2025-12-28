import streamlit as st
import re
import nltk
import joblib
from nltk.corpus import stopwords

# Load your trained model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Download stopwords (only needed the first time)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Map numeric predictions to emotion labels
emotion_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

# --- Text cleaning function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

# --- Prediction function ---
def predict_emotion(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    # Convert numeric label to emotion name
    emotion = emotion_map.get(int(prediction), "unknown")
    return emotion

# --- Streamlit UI ---
st.title("🧠 Mental Health Detection Chatbot")
st.write("Detects emotion from your text (Joy, Sadness, Anger, Fear)")

user_input = st.text_area("Enter your message here:")

if st.button("Detect Emotion"):
    if user_input.strip():
        emotion = predict_emotion(user_input)
        st.success(f"Detected Emotion: **{emotion.upper()}**")
    else:
        st.warning("Please enter a message to analyze.")


