import os
import cv2
import matplotlib.pyplot as plt
import argparse
from utils import draw_lane_lines

parser = argparse.ArgumentParser(description="Lane Detection")
parser.add_argument("--image", help="Path to the input image")
args = parser.parse_args()

image_path = args.image if args.image else "data_road/training/image_2/um_000000.png"
image = cv2.imread(image_path)

if image is None:
    print("Error: image not found. Check the path.")
else:
    original, lines_only, result = draw_lane_lines(image)

os.makedirs("output", exist_ok=True)
output_path = "output/kitti_result.png"
cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
print(f"Saved output to: {output_path}")

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(lines_only)
plt.title("Detected Lane Lines Only")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(result)
plt.title("Overlay Result")
plt.axis("off")

plt.show()
