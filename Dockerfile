FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

COPY flask_app/ /app/
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

RUN pip install -r requirements.txt

EXPOSE 8000  # Gunicorn default port

# CMD ["python", "app.py"]
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT:-8000}", "--timeout", "120", "app:app"]
