import cv2
import numpy as np


def detect_lesion(fusion_heatmap, original_image):

    # ----------------------------
    # Resize Heatmap
    # ----------------------------
    heatmap = cv2.resize(
        fusion_heatmap,
        (
            original_image.shape[1],
            original_image.shape[0]
        )
    )

    heatmap = heatmap.astype(np.float32)

    heatmap -= heatmap.min()

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    # ----------------------------
    # Adaptive Threshold
    # ----------------------------
    mean = np.mean(heatmap)

    std = np.std(heatmap)

    threshold = min(0.85, mean + std)

    mask = (heatmap >= threshold).astype(np.uint8) * 255

    # ----------------------------
    # Morphology
    # ----------------------------
    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    output = original_image.copy()

    lesion_area = 0

    lesion_percent = 0

    bbox = None

    if len(contours) > 0:

        largest = max(
            contours,
            key=cv2.contourArea
        )

        if cv2.contourArea(largest) > 100:

            lesion_area = cv2.contourArea(largest)

            total_area = (
                original_image.shape[0] *
                original_image.shape[1]
            )

            lesion_percent = (
                lesion_area /
                total_area
            ) * 100

            x, y, w, h = cv2.boundingRect(largest)

            bbox = (x, y, w, h)

            # Contour
            cv2.drawContours(
                output,
                [largest],
                -1,
                (0,255,0),
                3
            )

            # Bounding Box
            cv2.rectangle(
                output,
                (x,y),
                (x+w,y+h),
                (255,0,0),
                2
            )

            # Center Point
            cx = x + w // 2
            cy = y + h // 2

            cv2.circle(
                output,
                (cx,cy),
                5,
                (255,255,0),
                -1
            )

            cv2.putText(
                output,
                f"{lesion_percent:.2f}%",
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2
            )

    return (
        output,
        lesion_area,
        lesion_percent,
        bbox
    )