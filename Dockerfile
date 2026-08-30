# Use official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies needed for OpenCV, EasyOCR, and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR default English model weights into container cache
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

# Copy application files
COPY . .

# Expose default Streamlit port (8501) and FastAPI port (8000)
EXPOSE 8501 8000

# Default command: run Streamlit web app
# (Can be overridden in docker run or docker-compose to run api.py)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
