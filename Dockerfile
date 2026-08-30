# Use official Python lightweight image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install system dependencies for OpenCV, EasyOCR, and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (Hugging Face Spaces default UID 1000)
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# Copy requirements and install dependencies as non-root user
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download EasyOCR model weights into user cache
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

# Copy application files
COPY --chown=user:user . .

# Hugging Face Spaces listens on port 7860
EXPOSE 7860

# Launch Streamlit on port 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
