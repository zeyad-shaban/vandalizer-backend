FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    redis-server \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Step 2: Runtime defaults for Hugging Face Spaces
ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    DEBUG=false \
    RESET_JOBS_ON_STARTUP=true \
    JOB_TTL_SECONDS=3600 \
    JOB_CLEANUP_INTERVAL_SECONDS=300 \
    CELERY_BROKER_URL=redis://localhost:6379/0 \
    CELERY_RESULT_BACKEND=redis://localhost:6379/1

# Step 3. Download models for caching
RUN mkdir -p ./models
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('zeyadcode/ov_owlv2_model', local_dir='./models/ov_owlv2_model')"    
RUN python -c "from ultralytics import SAM; import os; os.chdir('./models'); SAM('mobile_sam.pt')"
RUN python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('google/owlv2-base-patch16-ensemble')"
RUN python -c "from simple_lama_inpainting import SimpleLama; SimpleLama()"
RUN python -c "from optimum.intel import OVStableDiffusionXLInpaintPipeline; pipe = OVStableDiffusionXLInpaintPipeline.from_pretrained('stabilityai/sdxl-turbo', export=True, device='CPU'); pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1); pipe.save_pretrained('./models/ov_sdxl_turbo_inpaint')"


# Step 4: Copy the rest of the application
COPY . .

RUN chmod +x start.sh

EXPOSE 7860

CMD ["./start.sh"]
