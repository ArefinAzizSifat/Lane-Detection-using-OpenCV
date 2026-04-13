import os
import cv2
import glob
import argparse
import csv
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

summary_rows = []
for image_path in image_paths:
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: could not read image {image_path}")
        continue

    original, lines_only, result, left_detected, right_detected = draw_lane_lines(
        image)
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

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Canny Edges")
    axes[1].axis("off")

    axes[2].imshow(roi_edges, cmap="gray")
    axes[2].set_title("ROI Edges")
    axes[2].axis("off")

    axes[3].imshow(result)
    axes[3].set_title("Overlay Result")
    axes[3].axis("off")

comparison_path = os.path.join(save_dir, f"{base_name}_comparison.png")
plt.tight_layout()
plt.savefig(comparison_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved comparison figure to: {comparison_path}")

summary_rows.append({
    "image_name": os.path.basename(image_path),
    "left_line_detected": left_detected,
    "right_line_detected": right_detected
})

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

    summary_path = "output/processing_summary.csv"
    os.makedirs("output", exist_ok=True)

    with open(summary_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["image_name", "left_line_detected",
                        "right_line_detected"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary to: {summary_path}")
total_images = len(summary_rows)
left_count = sum(row["left_line_detected"] for row in summary_rows)
right_count = sum(row["right_line_detected"] for row in summary_rows)
both_count = sum(
    row["left_line_detected"] and row["right_line_detected"]
    for row in summary_rows
)

print("\nBatch Processing Summary")
print(f"Total images processed: {total_images}")
print(f"Left line detected: {left_count}")
print(f"Right line detected: {right_count}")
print(f"Both lines detected: {both_count}")
