import cv2

image_path = "data_road/training/image_2/um_000000.png"

image = cv2.imread(image_path)

if image is None:
    print("Error: image not found. Check the path.")
else:
    print("Image loaded successfully.")
    print("Image shape:", image.shape)

    cv2.imshow("KITTI Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
