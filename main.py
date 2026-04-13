import os
import cv2
import matplotlib.pyplot as plt
import argparse
from utils import draw_lane_lines

parser = argparse.ArgumentParser(description="Lane Detection")
parser.add_argument(
    "--image",
    type=str,
    default="data_road/training/image_2/um_000000.png",
    help="Path to the input image"
)
parser.add_argument(
    "--output",
    type=str,
    default="output/kitti_result.png",
    help="Path to save the output image"
)
args = parser.parse_args()

image_path = args.image
output_path = args.output

image = cv2.imread(image_path)

if image is None:
    print("Error: image not found. Check the path.")
else:
    original, lines_only, result = draw_lane_lines(image)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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
