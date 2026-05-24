import asyncio
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def ensure_upload_dir() -> None:
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def reset_upload_dir() -> None:
    if config.UPLOAD_DIR.exists():
        shutil.rmtree(config.UPLOAD_DIR)
    ensure_upload_dir()


def create_job_dir(job_id: str) -> Path:
    ensure_upload_dir()
    job_path = config.UPLOAD_DIR / job_id
    job_path.mkdir()
    metadata = {
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(job_path / config.JOB_METADATA_NAME, "w") as f:
        json.dump(metadata, f, indent=2)
    return job_path


def _created_at_timestamp(job_path: Path) -> float:
    metadata_path = job_path / config.JOB_METADATA_NAME
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            created_at = datetime.fromisoformat(metadata["created_at"])
            return created_at.timestamp()
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            logger.warning("Ignoring invalid job metadata at %s", metadata_path)

    return job_path.stat().st_mtime


def cleanup_expired_jobs(
    ttl_seconds: int = config.JOB_TTL_SECONDS,
    now: float | None = None,
) -> int:
    ensure_upload_dir()
    now = time.time() if now is None else now
    deleted_count = 0

    for job_path in config.UPLOAD_DIR.iterdir():
        if not job_path.is_dir():
            continue

        try:
            age_seconds = now - _created_at_timestamp(job_path)
        except FileNotFoundError:
            continue

        if age_seconds < ttl_seconds:
            continue

        try:
            shutil.rmtree(job_path)
            deleted_count += 1
            logger.info("Deleted expired job folder %s", job_path)
        except FileNotFoundError:
            continue
        except OSError:
            logger.exception("Failed to delete expired job folder %s", job_path)

    return deleted_count


async def cleanup_expired_jobs_forever() -> None:
    while True:
        cleanup_expired_jobs()
        await asyncio.sleep(config.JOB_CLEANUP_INTERVAL_SECONDS)
