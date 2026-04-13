import cv2
import numpy as np


def region_of_interest(edges):
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
    return roi_edges


def process_image_steps(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    roi_edges = region_of_interest(edges)

    return gray, blur, edges, roi_edges


def make_line_points(y1, y2, line_params):
    if line_params is None:
        return None

    slope, intercept = line_params

    if slope == 0:
        return None

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return (x1, int(y1), x2, int(y2))


def draw_lane_lines(image):
    gray, blur, edges, roi_edges = process_image_steps(image)

    lines = cv2.HoughLinesP(
        roi_edges,
        2,
        np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=20
    )

    height = image.shape[0]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

    left_detected = left_line is not None
    right_detected = right_line is not None

    return image_rgb, line_image, combo_image, left_detected, right_detected
