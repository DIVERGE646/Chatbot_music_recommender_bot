from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import pickle
import json
import numpy as np
import random
import nltk
import os  # 🔹 ADDED

from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 🔹 Tell nltk to use the local nltk_data folder
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_path)

# Initialize Flask app
app = Flask(__name__)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()

# No need to re-download data here if you're using local `nltk_data`
# nltk.download('punkt')
# nltk.download('wordnet')

# Load chatbot model and data
model = load_model('model.keras')
intents = json.load(open('data.json'))
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Load music mood classifier model and tools
mood_classifier = pickle.load(open('mood_classifier.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# NLP preprocessing
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert sentence to bag of words
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Predict chatbot intent
def predict_class(sentence):
    bow_vector = bow(sentence, words)
    res = model.predict(np.array([bow_vector]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# Extract mood from user text (if used)
def extract_mood(text):
    mood_keywords = ['happy', 'sad', 'angry', 'relaxed', 'energetic', 'calm', 'bored']
    for mood in mood_keywords:
        if mood in text.lower():
            return mood
    return 'happy'  # Default

# Generate chatbot response
def get_response(intents_list, user_text):
    if not intents_list:
        return "I didn't understand. Try again."

    tag = intents_list[0]['intent']

    for intent in intents['intents']:
        if intent['tag'] == tag:
            # Special case for mood-based music recommendation
            if tag == "music_mood":
                detected_mood = extract_mood(user_text)
                try:
                    mood_encoded = label_encoder.transform([detected_mood])
                    mood_scaled = scaler.transform([mood_encoded])
                    prediction = mood_classifier.predict(mood_scaled)
                    predicted_genre = label_encoder.inverse_transform(prediction)[0]
                    return f"You're in a {detected_mood} mood. How about listening to some {predicted_genre} music?"
                except Exception as e:
                    return f"Sorry, I couldn't recognize your mood properly for a music suggestion."
            else:
                return random.choice(intent['responses'])

    return "Hmm, not sure about that."

# Web routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def chatbot_response():
    user_text = request.form['msg']
    intents_list = predict_class(user_text)
    response = get_response(intents_list, user_text)
    return jsonify({'response': response})

# Run app
if __name__ == '__main__':
    app.run(debug=True)
