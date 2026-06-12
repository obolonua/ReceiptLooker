import cv2
import numpy as np
import os
from pathlib import Path

# -----------------------------
# Paths and config
# -----------------------------
IMAGE_PATH = Path(__file__).resolve().parent.parent / "receipt" / "X00016469612.jpg"
OUTPUT_DIR = "segmented_lines"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load image
# -----------------------------
img = cv2.imread(str(IMAGE_PATH))

if img is None:
    raise Exception("Cannot load image")

original = img.copy()

# -----------------------------
# Preprocess image
# -----------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur a little so the threshold is less sensitive to sensor noise and compression artifacts.
gray = cv2.GaussianBlur(gray, (3, 3), 0)

# Convert text to white and background to black.
thresh = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31,
    15
)

# Remove small isolated pixels before grouping text into lines.
kernel_noise = np.ones((2, 2), np.uint8)

thresh = cv2.morphologyEx(
    thresh,
    cv2.MORPH_OPEN,
    kernel_noise
)

# Connect nearby characters horizontally so each receipt line becomes one blob.
kernel_line = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (35, 3)
)

connected = cv2.morphologyEx(
    thresh,
    cv2.MORPH_CLOSE,
    kernel_line
)

# Find external contours for each connected text line.
contours, _ = cv2.findContours(
    connected,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

line_boxes = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Filter tiny components that are almost certainly noise.
    if h < 8:
        continue

    if w < 20:
        continue

    area = w * h

    if area < 300:
        continue

    line_boxes.append((x, y, w, h))

# Sort lines from top to bottom.
line_boxes = sorted(line_boxes, key=lambda b: b[1])

# Merge boxes that belong to the same line when closing still split them.
merged = []

for box in line_boxes:
    x, y, w, h = box

    if len(merged) == 0:
        merged.append([x, y, w, h])
        continue

    px, py, pw, ph = merged[-1]

    center_prev = py + ph // 2
    center_curr = y + h // 2

    if abs(center_curr - center_prev) < 10:
        nx = min(px, x)
        ny = min(py, y)

        nr = max(px + pw, x + w)
        nb = max(py + ph, y + h)

        merged[-1] = [
            nx,
            ny,
            nr - nx,
            nb - ny
        ]
    else:
        merged.append([x, y, w, h])

# Draw line boxes on a copy of the original image for debugging.
visual = original.copy()

for idx, (x, y, w, h) in enumerate(merged):
    cv2.rectangle(
        visual,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        visual,
        str(idx),
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1
    )

# Save each detected line as a separate crop.
for idx, (x, y, w, h) in enumerate(merged):
    pad = 5

    y1 = max(0, y - pad)
    y2 = min(gray.shape[0], y + h + pad)

    line_img = original[y1:y2, :]

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            f"line_{idx:03d}.png"
        ),
        line_img
    )

print(f"Detected {len(merged)} lines")

# Show debug windows so the segmentation can be inspected interactively.
cv2.imshow("Threshold", thresh)
cv2.imshow("Connected", connected)
cv2.imshow("Lines", visual)

cv2.waitKey(0)
cv2.destroyAllWindows()
