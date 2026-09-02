import cv2
import numpy as np
import sys


def detect_plate_color(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]

    # Ignore outer edges because they often contain
    # car body, bumper, plate frame, etc.
    roi = image[
        int(h * 0.18): int(h * 0.82),
        int(w * 0.12): int(w * 0.88)
    ]

    # Mild denoising
    roi = cv2.GaussianBlur(roi, (5, 5), 0)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    total_pixels = roi.shape[0] * roi.shape[1]

    # ----------------------------
    # Define colour masks
    # ----------------------------

    # White / dirty white plate.
    # Threshold is kept slightly relaxed because
    # CCTV / compressed images can make white look grey.
    white_mask = (
        (S < 70) &
        (V > 80)
    )

    # Yellow
    yellow_mask = (
        (H >= 15) &
        (H <= 40) &
        (S > 60) &
        (V > 70)
    )

    # Green
    green_mask = (
        (H >= 35) &
        (H <= 95) &
        (S > 50) &
        (V > 50)
    )

    # Blue
    blue_mask = (
        (H >= 90) &
        (H <= 140) &
        (S > 50) &
        (V > 50)
    )

    # Red has two regions in HSV
    red_mask = (
        (
            (H >= 0) &
            (H <= 10)
        )
        |
        (
            (H >= 165) &
            (H <= 179)
        )
    ) & (S > 70) & (V > 50)

    # Black plate
    black_mask = V < 55

    scores = {
        "White": np.count_nonzero(white_mask) / total_pixels,
        "Yellow": np.count_nonzero(yellow_mask) / total_pixels,
        "Green": np.count_nonzero(green_mask) / total_pixels,
        "Blue": np.count_nonzero(blue_mask) / total_pixels,
        "Red": np.count_nonzero(red_mask) / total_pixels,
        "Black": np.count_nonzero(black_mask) / total_pixels,
    }

    detected_color = max(scores, key=scores.get)

    return detected_color, scores


if __name__ == "__main__":

    # if len(sys.argv) != 2:
    #     print("Usage:")
    #     print("python plate_color.py <plate_image>")
    #     sys.exit(1)

    # image_path = sys.argv[1]
    image_path = "plate4.png"

    color, scores = detect_plate_color(image_path)

    print(f"Plate Colour: {color}")

    print("\nColour scores:")

    for name, score in sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{name:<10}: {score:.2%}")