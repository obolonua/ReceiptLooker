import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
FONT_DIR = BASE_DIR / "data" / "fonts"
OUTPUT_DIR = BASE_DIR / "data" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 28
FONT_SIZE = 22
SAMPLES_PER_DIGIT = 2500

font_paths = list(FONT_DIR.glob("*.ttf")) + list(FONT_DIR.glob("*.ttc"))

if not font_paths:
    raise ValueError("No fonts found in data/fonts")


def apply_rotation(image):
    angle = random.uniform(-8, 8)
    return image.rotate(angle, fillcolor=255)


def apply_noise(image):
    array = np.array(image).astype(np.float32)
    noise = np.random.normal(0, 12, array.shape)
    array += noise
    array = np.clip(array, 0, 255)
    return Image.fromarray(array.astype(np.uint8))


def apply_blur(image):
    radius = random.uniform(0, 1.2)
    return image.filter(ImageFilter.GaussianBlur(radius))


def generate_digit_image(digit):
    image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=255)
    draw = ImageDraw.Draw(image)
    font_path = random.choice(font_paths)
    font = ImageFont.truetype(str(font_path), FONT_SIZE)
    x = random.randint(4, 10)
    y = random.randint(0, 6)
    draw.text((x, y), str(digit), font=font, fill=0)
    image = apply_rotation(image)
    image = apply_blur(image)
    image = apply_noise(image)
    return image


def create_dataset():
    X = []
    y = []

    for digit in range(10):
        print(f"Generating digit {digit}...")
        for _ in range(SAMPLES_PER_DIGIT):
            image = generate_digit_image(digit)
            array = np.array(image) / 255.0
            vector = array.flatten()
            X.append(vector)
            y.append(digit)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


if __name__ == "__main__":
    X, y = create_dataset()
    print("\nDataset created:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "X_val.npy", X_val)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_val.npy", y_val)

    print("\nDatasets saved in data/generated/")
