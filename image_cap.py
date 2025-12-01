# image_cap.py

"""
This module provides functionality for generating image captions
using a pre-trained BLIP image captioning model from Hugging Face.
"""

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load the pre-trained model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# ======================================================================
# image loading function
# ======================================================================

def load_image(image_path: str) -> Image.Image:
    """Load an image from a file path or URL.

    Args:
        image_path (str): The file path or URL of the image.

    Returns:
        PIL.Image.Image: The loaded image.
    """
    if image_path.startswith("http://") or image_path.startswith("https://"):
        image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return image


# ======================================================================
# caption generation functions
# ======================================================================

def generate_caption_from_pil(image: Image.Image) -> str:
    """Generate a caption for a given PIL image.

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        str: The generated caption.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def generate_caption(image_path: str) -> str:
    """Generate a caption for an image given its path or URL.

    Args:
        image_path (str): The file path or URL of the image.

    Returns:
        str: The generated caption.
    """
    image = load_image(image_path)
    return generate_caption_from_pil(image)


# Example usage (for quick testing from terminal)
if __name__ == "__main__":
    test_image = "https://huggingface.co/datasets/hf-internal-testing/sample-images/resolve/main/horse.png"
    print("Generated Caption:", generate_caption(test_image))

