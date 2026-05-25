import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageFilter


def get_prompt_list(prompt: str):
    return [txt.strip() for txt in prompt.split(".") if txt.strip() != ""]


def plot_groundingdino_boxes(img: Image.Image, result, figsize=(12, 8), show_scores=True):
    """
    img: PIL Image
    result: results[0] dict containing 'boxes', 'scores', 'labels'
    """
    img_np = np.array(img)

    plt.figure(figsize=figsize)
    plt.imshow(img_np)
    ax = plt.gca()

    boxes = result["boxes"]
    scores = result.get("scores", None)
    labels = result.get("labels", None)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)

        text = ""
        if labels is not None:
            text += str(labels[i])
        if show_scores and scores is not None:
            text += f" ({scores[i]:.2f})"

        if text:
            ax.text(x1, y1 - 5, text, color="red", fontsize=12, bbox=dict(facecolor="black", alpha=0.5, pad=2))

    plt.axis("off")
    plt.show()


MASK_OVERLAY_RGBA = (225, 29, 72, int(0.58 * 255))


def save_visual_mask_from_binary(mask, out_path):
    mask_bool = np.asarray(mask)
    if mask_bool.ndim == 3:
        mask_bool = np.any(mask_bool > 0, axis=0)
    elif mask_bool.dtype != bool:
        mask_bool = mask_bool > 0

    H, W = mask_bool.shape
    out = np.zeros((H, W, 4), dtype=np.uint8)
    out[mask_bool] = MASK_OVERLAY_RGBA
    Image.fromarray(out, mode="RGBA").save(out_path)


def save_visual_mask(mask_data, out_path):
    masks = np.asarray(mask_data)
    if masks.ndim == 2:
        combined_mask = masks > 0
    else:
        masks = (masks > 0.5) if masks.dtype != bool else masks.astype(bool)
        combined_mask = np.any(masks, axis=0) if masks.size else np.zeros(masks.shape[-2:], dtype=bool)

    save_visual_mask_from_binary(combined_mask, out_path)


def mask_image_to_binary(mask_img: Image.Image, size=None) -> Image.Image:
    rgba = mask_img.convert("RGBA")
    if size is not None and rgba.size != size:
        rgba = rgba.resize(size, Image.Resampling.NEAREST)

    rgba_arr = np.asarray(rgba)
    alpha = rgba_arr[..., 3]

    if np.any(alpha < 255):
        mask_bool = alpha > 12
    else:
        gray = np.asarray(rgba.convert("L"))
        mask_bool = gray > 127

    return Image.fromarray(mask_bool.astype(np.uint8) * 255, mode="L")


def blur_img_with_mask(
    original_img: Image.Image,
    mask_img: Image.Image,
    blur_radius: int = 15,
) -> Image.Image:
    original_image = original_img.convert("RGB")
    mask_image = mask_img.convert("L")

    # Create blurred version of image
    blurred_image = original_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    result = Image.composite(
        blurred_image,
        original_image,
        mask_img,
    )

    return result
