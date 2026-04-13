import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = "data_road/training/image_2/um_000000.png"

image = cv2.imread(image_path)

if image is None:
    print("Error: image not found. Check the path.")
else:
    print("Color image loaded successfully.")
    print("Color image shape:", image.shape)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Grayscale image shape:", gray.shape)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    print("Blurred image shape:", blur.shape)

    edges = cv2.Canny(blur, 50, 150)
    print("Edges image shape:", edges.shape)

    height, width = edges.shape

    polygon = np.array([[
        (100, height),
        (width - 100, height),
        (width // 2 + 80, height // 2 + 40),
        (width // 2 - 80, height // 2 + 40)
    ]], dtype=np.int32)

mask = np.zeros_like(edges)
cv2.fillPoly(mask, polygon, 255)

roi_edges = cv2.bitwise_and(edges, mask)

# OpenCV loads color as BGR, matplotlib expects RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_with_roi = image_rgb.copy()
cv2.polylines(image_with_roi, polygon, isClosed=True,
              color=(255, 0, 0), thickness=3)

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(image_with_roi)
plt.title("Original Image with ROI Polygon")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(roi_edges, cmap="gray")
plt.title("Edges After ROI")
plt.axis("off")

plt.show()
