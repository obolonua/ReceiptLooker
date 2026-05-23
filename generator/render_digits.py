import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
FONT_DIR = BASE_DIR / "data" / "fonts"

# -----------------------------
# Config
# -----------------------------

IMAGE_SIZE = 28
FONT_SIZE = 22

# -----------------------------
# Load Fonts
# -----------------------------

font_paths = list(FONT_DIR.glob("*.ttf")) + list(FONT_DIR.glob("*.ttc"))

if not font_paths:
    raise ValueError("No fonts found in data/fonts")

# -----------------------------
# Digit Generator
# -----------------------------


def generate_digit_image(digit):

    # White background
    image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=255)

    draw = ImageDraw.Draw(image)

    # Random font
    font_path = random.choice(font_paths)

    font = ImageFont.truetype(str(font_path), FONT_SIZE)

    # Random position
    x = random.randint(4, 10)
    y = random.randint(0, 6)

    # Draw digit
    draw.text(
        (x, y),
        str(digit),
        font=font,
        fill=0
    )

    # Apply augmentations

    image = apply_rotation(image)

    image = apply_blur(image)

    image = apply_noise(image)
    
    return image

# Rotate the image

def apply_rotation(image):

    angle = random.uniform(-8, 8)

    return image.rotate(
        angle,
        fillcolor=255
    )

# Add some noise to image
def apply_noise(image):

    array = np.array(image).astype(np.float32)

    noise = np.random.normal(
        loc=0,
        scale=5,
        size=array.shape
    )

    array += noise

    array = np.clip(array, 0, 255)

    return Image.fromarray(array.astype(np.uint8))

# Make the image blurry
def apply_blur(image):

    radius = random.uniform(0, 1.2)

    return image.filter(

        ImageFilter.GaussianBlur(radius=radius)

    )

# -----------------------------
# Visualization
# -----------------------------

if __name__ == "__main__":

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    for digit, ax in zip(range(10), axes.flatten()):

        image = generate_digit_image(digit)

        ax.imshow(image, cmap="gray")
        ax.set_title(f"Digit {digit}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()