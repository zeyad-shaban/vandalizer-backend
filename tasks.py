# %%
from celery import Celery
import config as config
import torch
from utils import get_prompt_list, plot_groundingdino_boxes, save_visual_mask, blur_img_with_mask
from PIL import Image
from transformers import BatchFeature
import numpy as np
import json
from typing import Any
from services import model_manager

MODELS: dict[str, Any] = {
    "detector": None,
    "detector_processor": None,
    "segmentor": None,
    "inpaintor": None,
    "remover": None,
}

celery_app = Celery(
    "worker",
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND,
)
celery_app.conf.update(
    broker_connection_retry_on_startup=True,
    result_expires=config.JOB_TTL_SECONDS,
    task_track_started=True,
)


@celery_app.task
def detect_objects(job_id: str, prompt: str) -> dict:
    prompt_list = get_prompt_list(prompt)

    job_path = config.UPLOAD_DIR / job_id
    detection_path = job_path / config.DETECTOR_OUT_PATH
    if detection_path.exists():
        detection_path.unlink()

    img = Image.open(job_path / config.INPUT_IMG_NAME)  # W x H

    model = model_manager.get_detector_model(MODELS)
    processor = model_manager.get_detector_processor(MODELS)

    inputs = processor(images=img, text=[prompt_list], return_tensors="pt")
    inputs = {name: t for name, t in inputs.items()}
    outputs = model(inputs)

    outputs = BatchFeature(
        {
            "logits": torch.tensor(outputs["logits"]),
            "pred_boxes": torch.tensor(outputs["pred_boxes"]),
        }
    )

    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=0.1,
        target_sizes=[img.size[::-1]],
    )

    result = results[0]
    result["scores"] = result["scores"].tolist()
    result["labels"] = result["labels"].tolist()
    result["boxes"] = result["boxes"].int().tolist()
    result["text_labels"] = [prompt_list[label] for label in result["labels"]]

    with open(detection_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


@celery_app.task(bind=True)
def segment_objects(self, job_id: str, bboxes=None, points=None, point_labels=None):
    job_path = config.UPLOAD_DIR / job_id
    img = Image.open(job_path / config.INPUT_IMG_NAME)

    model = model_manager.get_segmentor_model(MODELS)
    results = model(img, bboxes=bboxes, points=points, labels=point_labels)

    mask_data = results[0].masks.data.cpu().numpy()  # n_masks x H x W

    combined_mask = np.any(mask_data, axis=0).astype(np.uint8) * 255
    save_bin_path = job_path / config.SEGMENTOR_OUT_BIN_PATH
    save_visual_path = job_path / config.SEGMENTOR_OUT_VISUAL_PATH

    Image.fromarray(combined_mask).save(save_bin_path)
    save_visual_mask(mask_data, save_visual_path)

    if self.request.id is None:
        return results[0]

    return True


@celery_app.task(bind=True)
def inpaint(
    self,
    job_id: str,
    mode: str = "blur",
    positive_prompt: str = "",
    num_inference_steps: int = 4,
) -> bool:
    job_path = config.UPLOAD_DIR / job_id
    save_path = job_path / config.INPAINTOR_OUT_PATH
    if save_path.exists():
        save_path.unlink()

    orig_img = Image.open(job_path / config.INPUT_IMG_NAME).convert("RGB")
    mask_img = Image.open(job_path / config.SEGMENTOR_OUT_BIN_PATH).convert("L")

    if mode == "diffusion":
        model = model_manager.get_inpaintor_model(MODELS)
        model_size = (config.INPAINTOR_IMAGE_SIZE, config.INPAINTOR_IMAGE_SIZE)
        generated = model(
            prompt=positive_prompt,
            image=orig_img.resize(model_size),
            mask_image=mask_img.resize(model_size, Image.Resampling.NEAREST),
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
        ).images[0]
        result = generated.resize(orig_img.size)

    elif mode == "remove":
        model = model_manager.get_removing_model(MODELS)
        result = model(orig_img, mask_img)

    elif mode == "blur":
        result = blur_img_with_mask(orig_img, mask_img)

    else:
        raise ValueError(f"Unsupported inpaint mode: {mode}")

    result.convert("RGB").save(save_path)
        
    if self.request.id is None:
        return result
        
    return True


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    prompt = "head"
    img = Image.open(config.UPLOAD_DIR / config.DEBUG_JOB_ID / config.INPUT_IMG_NAME)

    # %%
    detection_res = detect_objects(job_id=config.DEBUG_JOB_ID, prompt=prompt)
    plot_groundingdino_boxes(img, detection_res)

    # %%
    segment_res = segment_objects.run(job_id=config.DEBUG_JOB_ID, bboxes=detection_res["boxes"])

    # %%
    inpainted_res = inpaint(
        job_id=config.DEBUG_JOB_ID,
        mode="diffusion",
        positive_prompt="red hat high quality",
        num_inference_steps=4,
    ) # type: ignore

    # %%
    img = segment_res.plot()
    plt.subplot(121)
    plt.imshow(img[:, :, ::-1])
    plt.axis("off")
    plt.show()
