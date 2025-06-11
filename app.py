from flask import Flask, request, jsonify
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
import string
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = Flask(__name__)

# Load model keras
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "ED_model.keras")
model = load_model(MODEL_PATH)

TOKENIZER_PATH = './tokenizer.json'
# Load tokenizer
with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
print("Tokenizer berhasil dimuat.")

MAX_SEQUENCE_LENGTH = 37  

# -------------------------------------------
# Function for Text Cleaning and Preprocessing
# -------------------------------------------
def cleaningText(text):
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

def casefoldingText(text):
    return text.lower()

slangwords = {
    "@": "di", "abis": "habis", "masi": "masih", "gua": "saya", "gw": "saya", "klo": "kalau",
    "afk": "tidak aktif", "thx": "terima kasih", "ty": "terima kasih", "tq": "terima kasih",
    "bgt": "banget", "maks": "maksimal", "gg": "hebat", "noob": "pemula", "poke": "pemula",
    "op": "terlalu kuat", "ez": "mudah", "dll": "dan lain lain",
}

def fix_slangwords(text):
    words = text.lower().split()
    fixed_words = [slangwords.get(word, word) for word in words]
    return ' '.join(fixed_words)

def tokenizingText(text):
    return word_tokenize(text)

def filteringText(words):
    factory = StopWordRemoverFactory()
    listStopwords = set(factory.get_stop_words())
    listStopwords1 = set(stopwords.words('english'))
    listStopwords.update(listStopwords1)
    listStopwords.update(['iya','yaa','gak','nya','na','sih','ku',"di","ga","ya","gaa","loh","kah","woi","woii","woy", "nge"])
    return [word for word in words if word not in listStopwords]

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemmingText(words):
    return [stemmer.stem(word) for word in words]

def full_preprocessing(text):
    text = cleaningText(text)
    text = casefoldingText(text)
    text = fix_slangwords(text)
    tokens = tokenizingText(text)
    filtered = filteringText(tokens)
    stemmed = stemmingText(filtered)
    return ' '.join(stemmed)

# -------------------------------------------
label_map = {0: 'anger', 1: 'joy', 2: 'neutral', 3: 'sadness'}

# ROOT ROUTE - TAMBAHKAN INI
@app.route("/")
def home():
    return jsonify({
        "message": "Emotion Detection API",
        "status": "API nya jalan",
        "endpoints": {
            "predict": "/predict (POST)",
            "method": "kirimkan JSON dengan key 'text' berisi kalimat yang ingin diprediksi emosi nya",
        }
    })

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()
#     if not data or 'text' not in data:
#         return jsonify({"error": "No text provided"}), 400

#     input_text = data['text']
#     cleaned_text = full_preprocessing(input_text)

#     # Tokenizing & Padding untuk input ke model
#     sequence = tokenizer.texts_to_sequences([cleaned_text])
#     padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

#     # Prediksi
#     prediction = model.predict(padded_sequence)
#     predicted_label = np.argmax(prediction)
#     predicted_emotion = label_map[predicted_label]

#     return jsonify({
#         "input": input_text,
#         "cleaned": cleaned_text,
#         "prediction": predicted_emotion,
#         "confidence": float(np.max(prediction))
#     })

# if __name__ == "__main__":
#     app.run(debug=True)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    user_id = data.get('user_id', None)
    input_text = data['text']
    
    if user_id is None:
        return jsonify({"error": "user_id is required"}), 400
    
    cleaned_text = full_preprocessing(input_text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction)
    predicted_emotion = label_map[predicted_label]

    return jsonify({
        "user_id": user_id,
        "input": input_text,
        "cleaned": cleaned_text,
        "prediction": predicted_emotion,
        "confidence": float(np.max(prediction))
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)