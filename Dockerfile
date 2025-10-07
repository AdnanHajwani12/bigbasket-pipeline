# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src ./src

# Create an empty models folder to avoid Docker copy errors
RUN mkdir -p ./models

# Set PYTHONPATH so Python can find your package
ENV PYTHONPATH="/app/src"

# Optional: default command (can be overridden)
CMD ["python", "-m", "src.bigbasket.train", "--outdir", "models"]
