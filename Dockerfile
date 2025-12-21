# Python 3.11 Slim Image
FROM python:3.11-slim

# Working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Requirements copy karein
COPY requirements.txt .

# Dependencies install karein (Numpy fix ke sath ta ke baad mein crash na ho)
RUN pip install --no-cache-dir -r requirements.txt

# Code copy karein
COPY . .

# NOTE: Humne yahan se 'RUN python ingest.py' HATA DIYA hai.
# Kyunke image banate waqt API Key nahi hoti.

# Port expose karein
EXPOSE 8501

# Command: Jab container chalega, tab pehle Ingest karega phir App chalayega
CMD ["sh", "-c", "python ingest.py && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
