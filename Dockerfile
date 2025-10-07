# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models

ENV PYTHONPATH="/app/src"

CMD ["python", "-m", "src.bigbasket.train"]