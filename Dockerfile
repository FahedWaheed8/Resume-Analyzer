FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model (optional, for future use)
# RUN python -m spacy download en_core_web_sm

# Copy application
COPY . .

# Create upload directory
RUN mkdir -p /tmp/uploads

EXPOSE 5000

ENV FLASK_ENV=production

CMD ["python", "app.py"]
