# MindBlown Emotion Detection

## Project Description
MindBlown Emotion Detection is a classification model in Natural Language Processing (NLP) for MindTracker feature in MindBlown App, designed to predict the emotions contained in a sentence, either when run locally or deployed through Hugging Face Spaces using Docker.

## Features
- Built on TensorFlow with Simple RNN and LSTM architectures.
- Designed to be lightweight and efficient, ensuring smooth performance both locally and on platforms like Hugging Face Spaces.
- Compatible with deployment using Docker, providing flexibility across various development and production environments.

---

## Running Locally

Follow the steps below to run the MindBlown Emotion Detection on your local machine using Docker:

### 1. Clone the Repository
```
git clone https://github.com/MindBlownDBS/Emotion-Detection.git
cd Emotion-Detection
```

### 2. Install Requirements (Recommended Using Python 3.11)
```
pip install -r requirements.txt
```

### 3. Build and Run Docker
```
docker build -t emotion-detection .
docker run -p 8080:8080 emotion-detection
```

## Deploy in Hugging Face Spaces

Follow the steps below to deploy your model in Hugging Face Spaces:

### 1. Clone the Repository
```
git clone https://github.com/MindBlownDBS/Emotion-Detection.git
cd Emotion-Detection
```
### 2. Create a Space on Hugging Face
- Visit Hugging Face Spaces
- Create a new Space and choose Docker as the runtime
- Upload all the files required for production in this repository.

 📂 Project Structure in Hugging Face Spaces
```
├── app.py
├── Dockerfile
├── requirements.txt
├── tokenizer.json
├── text_preprocessor.joblib
└── model/
```

### 3. Build and Deploy
Once everything is uploaded, Hugging Face will automatically build and run your container.