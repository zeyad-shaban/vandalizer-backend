---
title: Vandalizer Backend
emoji: 🖌️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Vandalizer Backend

FastAPI backend for Vandalizer, an AI image editing app for detecting, masking, and removing or replacing objects in images.

[Live app](https://vandalizer-frontend.vercel.app/) | [Hugging Face Space](https://huggingface.co/spaces/zeyadcode/vandalizer-backend) | [Demo video](https://youtu.be/dkpUaaLSOQU)

[![Watch the Vandalizer demo](https://img.youtube.com/vi/dkpUaaLSOQU/hqdefault.jpg)](https://youtu.be/dkpUaaLSOQU)

## Related Repositories

- [vandalizer-frontend](https://github.com/zeyad-shaban/vandalizer-frontend) - React/Vite UI for upload, mask editing, and result generation.
- [vandalizer-backend](https://github.com/zeyad-shaban/vandalizer-backend) - this FastAPI, Celery, Redis, and model inference service.
- [vandalizer-ai-workspace](https://github.com/zeyad-shaban/vandalizer-ai-workspace) - research notebooks and model prototyping workspace.

## What It Does

The backend owns the full image-processing pipeline. It accepts image uploads, stores each request as a job folder, runs long model tasks through Celery, exposes job status for polling, and serves generated artifacts back to the frontend.

Processing flow:

1. `POST /upload` saves the input image under a new UUID job folder.
2. Detection uses an OpenVINO-exported OWLv2 model with a Hugging Face processor.
3. Mask generation uses detected bounding boxes with MobileSAM.
4. Manual masks can be uploaded and normalized into binary and visual mask outputs.
5. Inpainting runs in one of three modes:
   - `blur` - masks are blurred with Pillow.
   - `remove` - object removal through SimpleLama.
   - `diffusion` - prompt-guided inpainting through an OpenVINO diffusion pipeline.
6. The frontend polls Celery status and downloads generated files from `/uploads`.

## Tech Stack

- Python 3.11
- FastAPI and Uvicorn
- Celery and Redis
- OpenVINO and Optimum Intel
- Transformers and Hugging Face Hub
- Ultralytics MobileSAM
- SimpleLama
- Pillow and OpenCV
- Docker and Hugging Face Spaces

## Project Structure

```text
main.py                     FastAPI app, routes, CORS, static uploads, lifecycle cleanup
tasks.py                    Celery tasks for detection, segmentation, mask generation, and inpainting
config.py                   Runtime paths, model names, Redis URLs, and job cleanup settings
utils.py                    Prompt parsing, mask conversion, visualization, and blur helpers
start.sh                    Container entrypoint for Redis, Celery, and Uvicorn
Dockerfile                  Hugging Face Spaces Docker image
requirements.txt            Python dependencies
schemas/
  inpaint_request.py        Pydantic request model for inpainting options
  segment_request.py        Pydantic request model for bounding boxes
services/
  model_manager.py          Lazy model loading and caching
  job_lifecycle.py          Job folder creation, metadata, expiration, and cleanup
```

## API Overview

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload an image and create a job |
| `GET` | `/job_status/{job_id}` | Read Celery task state |
| `POST` | `/process/detect_objects/{job_id}` | Run text-prompted object detection |
| `POST` | `/process/generate_mask/{job_id}` | Detect prompted objects and segment them into a mask |
| `POST` | `/process/segment_objects/{job_id}` | Segment provided bounding boxes |
| `POST` | `/api/upload-manual-mask` | Save a manually painted mask |
| `POST` | `/process/inpaint/{job_id}` | Generate the final edited image |
| `GET` | `/uploads/{job_id}/{filename}` | Serve generated job artifacts |

## Job Artifacts

Each upload creates a folder under `uploads/{job_id}`.

```text
input_img.png
job.json
detector_boxes.json
segmentor_masks_bin.png
segmentor_masks_visual.png
inpainted.png
```

`job.json` stores creation metadata so old jobs can be removed by the lifecycle cleanup task.

## Local Development

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Start Redis:

```bash
redis-server --bind 0.0.0.0 --protected-mode no
```

Start the Celery worker:

```bash
celery -A tasks worker --loglevel=info --pool=solo --concurrency=1
```

Start the API:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8080
```

Then point the frontend at:

```bash
VITE_API_BASE_URL=http://localhost:8080
```

## Docker

Build the image:

```bash
docker build -t vandalizer-backend .
```

Run the container:

```bash
docker run --rm -p 7860:7860 vandalizer-backend
```

The container entrypoint starts Redis, a Celery worker, and Uvicorn. It also clears runtime upload/output files on startup.

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEBUG` | `true` | Controls debug-oriented defaults |
| `UPLOAD_DIR` | `uploads` | Job artifact directory |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Celery broker |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/1` | Celery result backend |
| `INPAINTOR_MODEL_PATH` | `models/ov_sdxl_turbo_inpaint` | Local OpenVINO inpainting model path |
| `INPAINTOR_DEVICE` | `CPU` | OpenVINO target device |
| `JOB_TTL_SECONDS` | `3600` | Job artifact lifetime |
| `JOB_CLEANUP_INTERVAL_SECONDS` | `1500` | Cleanup loop interval |
| `RESET_JOBS_ON_STARTUP` | `not DEBUG` | Whether runtime jobs are cleared on startup |

## Deployment

This backend is designed to run as a Docker-based Hugging Face Space. The GitHub workflow in `.github/workflows/sync_to_hf.yml` syncs pushes from `main` to the Space repository.

The Docker image pre-downloads model assets so the Space can start with the required inference files already cached.

## Notes for Reviewers

The backend is intentionally separated from the frontend because model inference is slow and stateful compared with the browser UI. Celery gives the frontend a simple polling model while Redis tracks task state and FastAPI serves completed artifacts.
