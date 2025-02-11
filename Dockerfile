# Install necessary dependencies
RUN apt-get update && apt-get install -y libgomp1

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Download NLTK datasets at build time
RUN python -m nltk.downloader stopwords wordnet

# Copy application files
COPY flask_app/ /app/
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

# Expose correct port
EXPOSE 10000

# Start the Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT:-10000}", "--timeout", "120", "app:app"]


# CMD ["python", "app.py"]
# CMD ["gunicorn", "--bind", "0.0.0.0:${PORT:-8000}", "--timeout", "120", "app:app"]
# CMD /bin/sh -c "gunicorn --bind 0.0.0.0:${PORT:-10000} --timeout 120 app:app"
