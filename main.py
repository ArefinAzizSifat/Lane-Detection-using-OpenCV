import os
import cv2
import glob
import argparse
import matplotlib.pyplot as plt

from utils import draw_lane_lines, process_image_steps


parser = argparse.ArgumentParser(description="Lane Detection")
parser.add_argument(
    "--image",
    type=str,
    default="data_road/training/image_2/um_000000.png",
    help="Path to the input image or folder"
)
parser.add_argument(
    "--output",
    type=str,
    default="output/kitti_result.png",
    help="Path to save the output image or folder"
)
parser.add_argument(
    "--limit",
    type=int,
    default=5,
    help="Number of images to process from the folder"
)
parser.add_argument(
    "--show",
    action="store_true",
    help="Display matplotlib results"
)

args = parser.parse_args()

show_results = args.show
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
    gray, blur, edges, roi_edges = process_image_steps(image)

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

    # Save final overlay result
    cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Saved output to: {save_path}")

    # Save intermediate pipeline outputs
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    save_dir = os.path.dirname(save_path) if os.path.dirname(
        save_path) else "output"
    os.makedirs(save_dir, exist_ok=True)

    cv2.imwrite(os.path.join(save_dir, f"{base_name}_gray.png"), gray)
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_blur.png"), blur)
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_edges.png"), edges)
    cv2.imwrite(os.path.join(
        save_dir, f"{base_name}_roi_edges.png"), roi_edges)

    if show_results:
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
