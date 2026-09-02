import sys
import cv2
import numpy as np
from ultralytics import YOLO


# YOLO COCO model
model = YOLO("yolo11n-seg.pt")


def classify_color(bgr):
    """
    Convert an average BGR color into a simple human-readable vehicle color.
    """

    color = np.uint8([[bgr]])
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0][0]

    h, s, v = map(int, hsv)

    # Neutral colors first
    if v < 50:
        return "Black"

    if s < 35:
        if v > 200:
            return "White"
        elif v > 130:
            return "Silver / Light Grey"
        else:
            return "Grey"

    # Colored vehicles
    if h < 10 or h >= 170:
        return "Red"
    elif h < 22:
        return "Orange"
    elif h < 35:
        return "Yellow"
    elif h < 85:
        return "Green"
    elif h < 130:
        return "Blue"
    elif h < 160:
        return "Purple"
    else:
        return "Red"


def get_car_color(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    results = model(image, verbose=False)

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return None

    best_mask = None
    largest_area = 0

    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Consider car-like vehicles
        if class_name not in ["car"]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        area = (x2 - x1) * (y2 - y1)

        if area > largest_area:
            largest_area = area

            if result.masks is not None:
                best_mask = result.masks.data[i].cpu().numpy()

    if best_mask is None:
        return None

    # Resize segmentation mask to original image size
    mask = cv2.resize(
        best_mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    mask = mask > 0.5

    # Extract only car pixels
    car_pixels = image[mask]

    if len(car_pixels) == 0:
        return None

    # Remove extremely dark pixels, which are commonly
    # tyres, windows and deep shadows.
    hsv_pixels = cv2.cvtColor(
        car_pixels.reshape(-1, 1, 3),
        cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)

    valid = hsv_pixels[:, 2] > 50

    car_pixels = car_pixels[valid]

    if len(car_pixels) == 0:
        return None

    # Median is generally more resistant to reflections
    # and highlights than simple mean.
    median_bgr = np.median(car_pixels, axis=0).astype(np.uint8)

    return classify_color(median_bgr)


if __name__ == "__main__":

    # if len(sys.argv) != 2:
    #     print("Usage: python car_color.py <image_path>")
    #     sys.exit(1)

    # image_path = sys.argv[1]
    image_path = "AR11.jpg"

    color = get_car_color(image_path)

    if color:
        print(f"Car Colour: {color}")
    else:
        print("Could not detect a car in the image.")
