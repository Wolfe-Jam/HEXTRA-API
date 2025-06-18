# Use Python slim image with pre-installed dependencies
FROM python:3.11-slim
# This dummy line forces a rebuild. Change or remove later.

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Good for documentation)
EXPOSE 8000

# Run the application -- HARDCODING PORT FOR DEBUGGING
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]