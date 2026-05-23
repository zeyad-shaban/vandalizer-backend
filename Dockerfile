FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    redis-server \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 2: Enable fast downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Step 3. Download models for caching
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('zeyadcode/ov_owlv2_model', local_dir='./models/ov_owlv2_model')"    
RUN python -c "from ultralytics import SAM; import os; os.chdir('./models'); SAM('mobile_sam.pt')"
RUN python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('google/owlv2-base-patch16-ensemble')"

# Step 4: Copy the rest of the application
COPY . .

RUN chmod +x start.sh

EXPOSE 7860

CMD ["./start.sh"]