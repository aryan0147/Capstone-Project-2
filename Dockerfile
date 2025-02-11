#  Use a minimal Python base image
FROM python:3.10-slim

#  Set working directory inside the container
WORKDIR /app

#  Install necessary system dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

#  Copy requirements first to leverage Docker cache
COPY requirements.txt /app/

# Install dependencies (no-cache to keep image small)
RUN pip install --no-cache-dir -r requirements.txt

#  Download NLTK datasets at build time
RUN python -m nltk.downloader -d /app/nltk_data stopwords wordnet

#  Copy application files after dependencies are installed
COPY flask_app/ /app/
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

# Expose correct port (Render might use a dynamic port)
EXPOSE 10000

#  Run Gunicorn properly
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--timeout", "120", "app:app"]