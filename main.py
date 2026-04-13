import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = "data_road/training/image_2/um_000000.png"

image = cv2.imread(image_path)


def make_line_points(y1, y2, line_params):
    if line_params is None:
        return None

    slope, intercept = line_params

    if slope == 0:
        return None

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return (x1, int(y1), x2, int(y2))


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

lines = cv2.HoughLinesP(
    roi_edges,
    2,
    np.pi / 180,
    threshold=50,
    minLineLength=40,
    maxLineGap=20
)

# OpenCV loads color as BGR, matplotlib expects RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_with_roi = image_rgb.copy()

line_image = image_rgb.copy()
cv2.polylines(image_with_roi, polygon, isClosed=True,
              color=(255, 0, 0), thickness=3)
if lines is not None:
    print("Number of detected lines:", len(lines))
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
else:
    print("No lines detected.")

if lines is not None:
    left_lines = []
    right_lines = []

left_image = image_rgb.copy()
right_image = image_rgb.copy()

if lines is not None:
    print("Total detected lines:", len(lines))

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)

        if abs(slope) > 0.5:
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
                cv2.line(left_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            else:
                right_lines.append((x1, y1, x2, y2))
                cv2.line(right_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    print("Left lines:", len(left_lines))
    print("Right lines:", len(right_lines))
else:
    print("No lines detected.")

left_fits = []
right_fits = []

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x1 == x2:
            continue

        slope = (y2 - y1) / (x2 - x1)

        if abs(slope) > 0.5:
            intercept = y1 - slope * x1

            if slope < 0:
                left_fits.append((slope, intercept))
            else:
                right_fits.append((slope, intercept))

left_avg = np.mean(left_fits, axis=0) if left_fits else None
right_avg = np.mean(right_fits, axis=0) if right_fits else None

y1 = height
y2 = int(height * 0.6)

left_line = make_line_points(y1, y2, left_avg)
right_line = make_line_points(y1, y2, right_avg)

line_image = np.zeros_like(image_rgb)

if left_line is not None:
    x1, y1, x2, y2 = left_line
    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 8)

if right_line is not None:
    x1, y1, x2, y2 = right_line
    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 8)

combo_image = cv2.addWeighted(image_rgb, 0.8, line_image, 1.0, 0)

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(line_image)
plt.title("Detected Lane Lines Only")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(combo_image)
plt.title("Overlay Result")
plt.axis("off")

plt.show()
