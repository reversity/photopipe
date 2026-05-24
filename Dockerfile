FROM python:3.12-slim

# Install system dependencies.
#
# Note: Docker is intended for the curate/finalize pipeline (processing
# already-scanned images). USB/network scanner access from inside a Linux
# container on macOS is fragile; for the capture phase, install PhotoPipe
# directly on the host via install-standalone.sh.
RUN apt-get update && apt-get install -y \
    libexif-dev \
    perl \
    libimage-exiftool-perl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Create directories for data persistence
RUN mkdir -p /data/input /data/output /data/archive /data/config

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV PHOTOPIPE_DATA_DIR=/data

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the app
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
