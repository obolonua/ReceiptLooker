import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

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

    return image


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