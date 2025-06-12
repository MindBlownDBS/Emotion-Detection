FROM python:3.11-slim

WORKDIR /app

# Install build tools & dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for cache and data directories
ENV HF_HOME=/tmp/.cache/huggingface
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV NLTK_DATA=/tmp/nltk_data

RUN mkdir -p $HF_HOME $MPLCONFIGDIR $NLTK_DATA
RUN chmod -R 777 $MPLCONFIGDIR $NLTK_DATA $HF_HOME


# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK resources (punkt, stopwords)
RUN python -m nltk.downloader -d /tmp/nltk_data punkt stopwords

# Copy project files
COPY . .

EXPOSE 6000

CMD ["python", "app.py"]