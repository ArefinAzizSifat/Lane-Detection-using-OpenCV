import os
import cv2
import matplotlib.pyplot as plt
import argparse
import glob
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
parser.add_argument(
    "--limit",
    type=int,
    default=5,
    help="Number of images to process from the folder"
)
args = parser.parse_args()

image_input = args.image
output_path = args.output
limit = args.limit

if os.path.isdir(image_input):
    image_paths = sorted(glob.glob(os.path.join(image_input, "*.png")))[:limit]
else:
    image_paths = [image_input]

for image_path in image_paths:
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: could not read image {image_path}")
        continue

    original, lines_only, result = draw_lane_lines(image)

    if os.path.isdir(image_input):
        output_dir = output_path
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.basename(image_path)
        save_path = os.path.join(output_dir, filename)
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        save_path = output_path

    cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Saved output to: {save_path}")

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
