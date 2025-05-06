FROM --platform=arm64 python:3.11-slim-bookworm

# Install packages
RUN apt-get update && apt-get install -y build-essential

RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt
# Download small NER model locally
RUN python3 -m spacy download en_core_web_sm

# RUN huggingface_cli login

COPY . .

ENV PYTHONPATH="/app/"
ENV FLASK_APP=manage.py
EXPOSE 5000



