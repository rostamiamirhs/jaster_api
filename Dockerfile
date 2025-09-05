FROM python:3.11-slim

# Install system dependencies for geospatial libraries
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    libexpat1-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
# Set environment variables for GDAL
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data and results
RUN mkdir -p data results static

# Expose port
EXPOSE 8080

# Start the application
CMD ["uvicorn", "simple_test:app", "--host", "0.0.0.0", "--port", "8080"]