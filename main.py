import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import uuid
import shutil
import tasks as tasks
from celery.result import AsyncResult
import config as config
from schemas import InpaintRequest, SegmentRequest
from services import job_lifecycle
from PIL import Image, UnidentifiedImageError
import numpy as np
from utils import mask_image_to_binary, save_visual_mask_from_binary

job_lifecycle.ensure_upload_dir()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.RESET_JOBS_ON_STARTUP:
        job_lifecycle.reset_upload_dir()

    cleanup_task = asyncio.create_task(job_lifecycle.cleanup_expired_jobs_forever())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Vandalizer", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=config.UPLOAD_DIR), name="uploads")


@app.get("/")
def health_check():
    return {"status": "running"}


@app.post("/upload")
def upload(img: UploadFile = File(...)):
    job_id = str(uuid.uuid4())  # uuid1 exposes mac address and time, uuid3,5 uses a key value to generate a hash, same input gets same out, uuid4 is random
    job_folder = job_lifecycle.create_job_dir(job_id)

    save_path = job_folder / config.INPUT_IMG_NAME
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(img.file, buffer)

    return {"job_id": job_id}
    
@app.get("/job_status/{job_id}")
def get_job_status(job_id: str):
    result = AsyncResult(job_id, app=tasks.celery_app)
    return {
        "job_id": job_id,
        "status": result.status,
    }


# putting default signature as Form(...) or File(...) makes it a form data, otherwise it would be a query data
@app.post("/process/detect_objects/{job_id}")
async def detect_objects(job_id: str, prompt: str = Form(...)):
    tasks.celery_app.backend.delete(f"celery-task-meta-{job_id}")
    tasks.detect_objects.apply_async(args=[job_id, prompt], task_id=job_id)
    # tasks.detect_objects(prompt=prompt, job_id=job_id)
    return job_id


@app.post("/process/generate_mask/{job_id}")
async def generate_mask(job_id: str, prompt: str = Form(...)):
    tasks.celery_app.backend.delete(f"celery-task-meta-{job_id}")
    tasks.generate_mask.apply_async(args=[job_id, prompt], task_id=job_id)
    return job_id


@app.post("/process/segment_objects/{job_id}")
async def segment_objects(job_id: str, data: SegmentRequest):
    tasks.celery_app.backend.delete(f"celery-task-meta-{job_id}")
    tasks.segment_objects.apply_async(args=[job_id, data.bboxes], task_id=job_id)
    return job_id


@app.post("/api/upload-manual-mask")
async def upload_manual_mask(job_id: str = Form(...), mask: UploadFile = File(...)):
    job_path = config.UPLOAD_DIR / job_id
    input_path = job_path / config.INPUT_IMG_NAME

    if not job_path.exists() or not input_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        mask_img = Image.open(mask.file)
        mask_img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid mask image") from exc

    input_img = Image.open(input_path)
    binary_mask = mask_image_to_binary(mask_img, size=input_img.size)
    binary_mask.save(job_path / config.SEGMENTOR_OUT_BIN_PATH)
    save_visual_mask_from_binary(
        np.asarray(binary_mask) > 0,
        job_path / config.SEGMENTOR_OUT_VISUAL_PATH,
    )

    return {"job_id": job_id, "mask_uploaded": True}


@app.post("/process/inpaint/{job_id}")
async def inpaint(job_id: str, data: InpaintRequest):
    tasks.celery_app.backend.delete(f"celery-task-meta-{job_id}")
    tasks.inpaint.apply_async(
        args=[
            job_id,
            data.mode,
            data.positive_prompt,
            data.strength,
            data.num_inference_steps,
        ],
        task_id=job_id,
    )
    return job_id


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True, reload_delay=1)
