FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/bin:$PATH"

# Cài đặt Python và các công cụ cơ bản
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Tạo symlink cho python
RUN ln -s /usr/bin/python3.11 /usr/local/bin/python && \
    ln -s /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /app

# Copy thư mục source code
COPY . /app/

# Cài đặt thư viện Python
# Do chúng ta dùng vLLM, cần cài đặt pytorch chuẩn CUDA 12.1
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

# Tạo thư mục chứa dữ liệu nếu chưa có
RUN mkdir -p /app/data/models /app/data/results /app/data/indexes

# (Tùy chọn) Chạy script tải model offline khi build image
# RUN python scripts/download_models.py

# Entrypoint mặc định (Sẽ truyền số query vào qua tham số docker run)
ENTRYPOINT ["python", "scripts/run_submission_local.py"]
