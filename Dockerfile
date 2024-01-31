# Base CUDA image
FROM cnstark/pytorch:2.0.1-py3.9.17-cuda11.8.0-ubuntu20.04

LABEL maintainer="icubic3@gmail.com"
LABEL version="dev-20240127-api"
LABEL description="Docker image for GPT-SoVITS API"

# Install 3rd party apps
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata ffmpeg libsox-dev parallel aria2 git git-lfs && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

# Copy application
WORKDIR /workspace
COPY . /workspace

# install python packages
RUN pip install -r requirements.txt

# Download models
RUN chmod +x /workspace/Docker/download.sh && /workspace/Docker/download.sh

# Set environment variable for NLTK data
ENV NLTK_DATA /workspace/nltk_data

CMD ["python", "./tts_api.py"]
