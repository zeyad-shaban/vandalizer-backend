from transformers import AutoProcessor
from ultralytics import SAM
import openvino as ov
import config as config
from optimum.intel import OVStableDiffusionXLInpaintPipeline
from simple_lama_inpainting import SimpleLama


def get_detector_model(MODELS: dict):
    if MODELS["detector"] is None:
        core = ov.Core()
        model = core.read_model(config.DETECTOR_MODEL_PATH)

        MODELS["detector"] = core.compile_model(model, "CPU")

    return MODELS["detector"]


def get_detector_processor(MODELS: dict):
    if MODELS["detector_processor"] is None:
        MODELS["detector_processor"] = AutoProcessor.from_pretrained(config.DETECTOR_MODEL_NAME, use_fast=True)
    return MODELS["detector_processor"]


def get_segmentor_model(MODELS: dict):
    if MODELS["segmentor"] is None:
        MODELS["segmentor"] = SAM(config.SEGMENTOR_MODEL_NAME)
    return MODELS["segmentor"]


def get_inpaintor_model(MODELS: dict):
    if MODELS["inpaintor"] is None:
        model_source = config.INPAINTOR_MODEL_PATH if config.INPAINTOR_MODEL_PATH.exists() else config.INPAINTOR_MODEL_NAME
        kwargs = {"device": config.INPAINTOR_DEVICE}
        if model_source == config.INPAINTOR_MODEL_NAME:
            kwargs["export"] = True

        pipe = OVStableDiffusionXLInpaintPipeline.from_pretrained(
            str(model_source),
            **kwargs,
        )
        pipe.reshape(
            batch_size=1,
            height=config.INPAINTOR_IMAGE_SIZE,
            width=config.INPAINTOR_IMAGE_SIZE,
            num_images_per_prompt=1,
        )
        pipe.compile()
        MODELS["inpaintor"] = pipe
    return MODELS["inpaintor"]


def get_removing_model(MODELS: dict):
    if MODELS["remover"] is None:
        import torch

        # Intercept the load function to force CPU mapping
        original_load = torch.jit.load
        torch.jit.load = lambda *a, **kw: original_load(*a, **{**kw, "map_location": "cpu"})

        MODELS["remover"] = SimpleLama()

        # Restore the original PyTorch load function right after
        torch.jit.load = original_load

    return MODELS["remover"]
