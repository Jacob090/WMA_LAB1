import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LAB1 - Detekcja i śledzenie czerwonego obiektu w materiale wideo."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Ścieżka do pliku wideo, np. sample.mp4",
    )
    return parser.parse_args()


def open_video(path_str: str) -> cv2.VideoCapture:
    path = Path(path_str)

    if not path.is_file():
        print(f"[BŁĄD] Nie znaleziono pliku wideo: {path}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[BŁĄD] Nie udało się otworzyć pliku wideo: {path}", file=sys.stderr)
        sys.exit(1)

    return cap


def segment_red_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Zakresy czerwieni w HSV
    lower_red_1 = np.array([0, 120, 70], dtype=np.uint8)
    upper_red_1 = np.array([5, 255, 255], dtype=np.uint8)

    lower_red_2 = np.array([170, 120, 70], dtype=np.uint8)
    upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    mask = cv2.bitwise_or(mask1, mask2)
    return mask



def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Oczyszczanie maski z szumów oraz wypełnianie dziur
    za pomocą operacji morfologicznych:
    - opening (erozja + dylatacja) usuwa drobne plamki,
    - closing (dylatacja + erozja) wypełnia dziury.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


def find_largest_contour(mask: np.ndarray, min_area: float = 300.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    return largest


def compute_centroid(contour) -> tuple[int, int] | None:
    moments = cv2.moments(contour)
    m00 = moments.get("m00", 0.0)
    if m00 == 0:
        return None
    cx = int(moments["m10"] / m00)
    cy = int(moments["m01"] / m00)
    return cx, cy


def draw_deviation_bars(
    frame: np.ndarray, deviation_px: int, max_bar_fraction: float = 0.4
) -> None:
    h, w = frame.shape[:2]
    x_center = w // 2

    # środek kadru
    cv2.line(frame, (x_center, 0), (x_center, h), (255, 255, 255), 1)

    max_bar_len = int(w * max_bar_fraction)
    # Normalizacja odchylenia do zakresu [-1, 1]
    norm = np.clip(deviation_px / max(w / 2, 1), -1.0, 1.0)
    bar_len = int(abs(norm) * max_bar_len)

    bar_height = 20
    y_top = 10
    y_bottom = y_top + bar_height

    if deviation_px > 0:
        start_x = x_center
        end_x = x_center + bar_len
        color = (0, 255, 0)
    elif deviation_px < 0:
        start_x = x_center - bar_len
        end_x = x_center
        color = (0, 255, 255)
    else:
        start_x = end_x = x_center
        color = (128, 128, 128)

    if start_x != end_x:
        cv2.rectangle(frame, (start_x, y_top), (end_x, y_bottom), color, thickness=-1)

    # Informacja liczbowa.
    if deviation_px > 0:
        direction = "prawo"
    elif deviation_px < 0:
        direction = "lewo"
    else:
        direction = "brak"

    text = f"Odchylenie: {deviation_px:+d} px ({direction})"
    cv2.putText(
        frame,
        text,
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def process_video(cap: cv2.VideoCapture) -> None:
    window_original = "Oryginalne wideo"
    window_mask = "Maska (HSV + morfologia)"

    cv2.namedWindow(window_original, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_mask, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask_raw = segment_red_hsv(frame)
        mask_clean = clean_mask(mask_raw)

        contour = find_largest_contour(mask_clean)
        h, w = frame.shape[:2]
        x_center = w // 2
        deviation_px = 0

        if contour is not None:
            centroid = compute_centroid(contour)
            if centroid is not None:
                cx, cy = centroid
                deviation_px = cx - x_center

                (x_circle, y_circle), radius = cv2.minEnclosingCircle(contour)
                center_circle = (int(x_circle), int(y_circle))
                radius_int = int(radius)

                cv2.circle(frame, center_circle, radius_int, (0, 255, 0), 2)
                cv2.circle(frame, centroid, 4, (0, 0, 255), -1)

        draw_deviation_bars(frame, deviation_px)

        cv2.imshow(window_original, frame)
        cv2.imshow(window_mask, mask_clean)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    cap = open_video(args.video)
    process_video(cap)


if __name__ == "__main__":
    main()