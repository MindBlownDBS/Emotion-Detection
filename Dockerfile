FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download nltk data saat build
RUN python -m nltk.downloader punkt_tab stopwords

COPY . .

EXPOSE 6000
CMD ["python", "app.py"]
