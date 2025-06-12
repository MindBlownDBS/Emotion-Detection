from flask import Flask, request, jsonify
import os

# -------------------------------
# Setup environment variables BEFORE other imports
# -------------------------------
os.environ['HF_HOME'] = '/tmp/.cache/huggingface'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['NLTK_DATA'] = '/tmp/nltk_data'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import nltk
nltk.data.path.append(os.environ['NLTK_DATA'])
nltk.download('punkt', download_dir=os.environ['NLTK_DATA'])
nltk.download('stopwords', download_dir=os.environ['NLTK_DATA'])

import json
import numpy as np
import joblib
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# -------------------------------
# Class TextPreprocessor
# -------------------------------
class TextPreprocessor:
    def __init__(self, slangwords=None):
        self.slangwords = slangwords or {}

        factory = StopWordRemoverFactory()
        self.stopwords = set(factory.get_stop_words())
        self.stopwords.update(stopwords.words('english'))
        self.stopwords.update(['iya', 'yaa', 'gak', 'nya', 'na', 'sih', 'ku', "di", "ga", "ya", "gaa", "loh", "kah", "woi", "woii", "woy", "nge"])

        stem_factory = StemmerFactory()
        self.stemmer = stem_factory.create_stemmer()

    def cleaning_text(self, text):
        emoticon_pattern = re.compile(
            "["u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        
        text = emoticon_pattern.sub(r'', text)
        text = re.sub(r'@[A-Za-z0-9]+', '', text)
        text = re.sub(r'#[A-Za-z0-9]+', '', text)
        text = re.sub(r'RT[\s]', '', text)
        text = re.sub(r"http\S+", '', text)
        text = re.sub(r'[0-9]+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        text = text.replace('\n', ' ')
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.strip()
        return text

    def fix_slangwords(self, text):
        words = text.lower().split()
        fixed_words = [self.slangwords.get(word, word) for word in words]
        return ' '.join(fixed_words)

    def preprocess(self, text):
        text = self.cleaning_text(text)
        text = text.lower()
        text = self.fix_slangwords(text)
        tokens = word_tokenize(text)
        filtered = [word for word in tokens if word not in self.stopwords]
        stemmed = [self.stemmer.stem(word) for word in filtered]
        return stemmed

    def transform(self, texts):
        return [self.preprocess(text) for text in texts]


# -------------------------------
# Load Model, Tokenizer, Preprocessor
# -------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "ED_model.keras")
model = load_model(MODEL_PATH)

TOKENIZER_PATH = './tokenizer.json'
with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
    tokenizer = tokenizer_from_json(json.load(f))
print("Tokenizer berhasil dimuat.")

PREPROCESSOR_PATH = './text_preprocessor.joblib'
preprocessor = joblib.load(PREPROCESSOR_PATH)
print("Preprocessor berhasil dimuat.")

MAX_SEQUENCE_LENGTH = 37
label_map = {0: 'anger', 1: 'joy', 2: 'neutral', 3: 'sadness'}

# -------------------------------
# Endpoints
# -------------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "Emotion Detection API",
        "status": "API is running",
        "endpoints": {
            "predict": "/predict (POST)",
            "method": "Send JSON with key 'text' containing the sentence to predict its emotion"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    input_text = data['text']
    cleaned_tokens = preprocessor.transform([input_text])[0]
    cleaned_text = ' '.join(cleaned_tokens)

    if not cleaned_text.strip():
        return jsonify({"error": "Input text is empty after preprocessing"}), 400

    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction)
    predicted_emotion = label_map[predicted_label]

    return jsonify({
        "input": input_text,
        "cleaned": cleaned_text,
        "prediction": predicted_emotion,
        "confidence": float(np.max(prediction))
    })


# -------------------------------
# Run the App
# -------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)
