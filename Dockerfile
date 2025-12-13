FROM python:3.10-slim

# Installing necessary packages
RUN apt update
RUN apt install -y dos2unix
RUN apt install -y wget
RUN apt install -y unzip

# Setting working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
-f https://download.pytorch.org/whl/torch_stable.html

# Copy source code and notebooks
COPY src/ src/
COPY notebook/ notebook/
COPY run.sh run.sh

# Create necessary directories
RUN mkdir -p /app/data
RUN mkdir -p /app/intermediate
RUN mkdir -p /app/output

# Fix line endings and make executable
RUN dos2unix /app/run.sh
RUN chmod +x /app/run.sh || true

# Running training script by default
CMD ["bash", "/app/run.sh"]
