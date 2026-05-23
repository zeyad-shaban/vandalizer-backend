from pathlib import Path

DEBUG = True
DEBUG_JOB_ID = "debug-id"

UPLOAD_DIR = Path("uploads")
INPUT_IMG_NAME = "input_img.png"
DETECTOR_MODEL_PATH = "./models/ov_owlv2_model/ov_owlv2_model.xml"
DETECTOR_MODEL_NAME = "google/owlv2-base-patch16-ensemble"
SEGMENTOR_MODEL_NAME = "models/mobile_sam.pt"

SEGMENTOR_OUT_BIN_PATH = "segmentor_masks_bin.png"
SEGMENTOR_OUT_VISUAL_PATH = "segmentor_masks_visual.png"

DETECTOR_OUT_PATH = "detector_boxes.json"

INPAINTOR_OUT_PATH = "inpainted.png"

INPAINTOR_MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"