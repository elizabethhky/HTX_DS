# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy your app code
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install fastapi uvicorn[standard] transformers torch torchaudio pydub numpy python-multipart

# Expose port
EXPOSE 8001

# Run the app
CMD ["uvicorn", "asr_api:app", "--host", "0.0.0.0", "--port", "8001"]
