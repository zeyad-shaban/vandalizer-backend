import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEBUG = _env_bool("DEBUG", True)
DEBUG_JOB_ID = "debug-id"

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
INPUT_IMG_NAME = "input_img.png"
DETECTOR_MODEL_PATH = "./models/ov_owlv2_model/ov_owlv2_model.xml"
DETECTOR_MODEL_NAME = "google/owlv2-base-patch16-ensemble"
SEGMENTOR_MODEL_NAME = "models/mobile_sam.pt"

SEGMENTOR_OUT_BIN_PATH = "segmentor_masks_bin.png"
SEGMENTOR_OUT_VISUAL_PATH = "segmentor_masks_visual.png"

DETECTOR_OUT_PATH = "detector_boxes.json"

INPAINTOR_OUT_PATH = "inpainted.png"

INPAINTOR_MODEL_NAME = "stabilityai/sdxl-turbo"
INPAINTOR_MODEL_PATH = Path(os.getenv("INPAINTOR_MODEL_PATH", "models/ov_sdxl_turbo_inpaint"))
INPAINTOR_DEVICE = os.getenv("INPAINTOR_DEVICE", "CPU")
INPAINTOR_IMAGE_SIZE = 512

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

JOB_METADATA_NAME = "job.json"
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))
JOB_CLEANUP_INTERVAL_SECONDS = int(os.getenv("JOB_CLEANUP_INTERVAL_SECONDS", "1500"))
RESET_JOBS_ON_STARTUP = _env_bool("RESET_JOBS_ON_STARTUP", not DEBUG)
