import matplotlib.pyplot as plt
import colorsys
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


def save_visual_mask(mask_data, out_path):
    masks = (mask_data > 0.5) if mask_data.dtype != bool else mask_data.astype(bool)

    n_masks, H, W = masks.shape
    out = np.zeros((H, W, 4), dtype=np.uint8)
    alpha_val = int(0.7 * 255)

    # generate distinct colors via HSV
    for i in range(n_masks):
        h = (i / n_masks) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)

        m = masks[i]
        if m.dtype != bool:
            m = m.astype(bool)

        # apply color and alpha (later masks overwrite earlier ones)
        out[m, 0] = r
        out[m, 1] = g
        out[m, 2] = b
        out[m, 3] = alpha_val

    # optional: ensure background alpha = 0 (already zero by initialization)
    img = Image.fromarray(out, mode="RGBA")
    img.save(out_path)


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
