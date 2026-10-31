import os
import cv2
import joblib
import time
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2

from src.preprocess import preprocess
from src.xgboost_predict import extract_xgb_features
from src.scorecam import generate_scorecam, overlay_heatmap
from src.lesion_detection import detect_lesion
from src.report import generate_report


# ====================================================
# LOAD CNN MODELS
# ====================================================

print("Loading CNN models...")

model1 = tf.keras.models.load_model(
    "models/oral_cancer_model.keras"
)

model2 = tf.keras.models.load_model(
    "models/oral_cancer_mobilenet-1.keras"
)

SIZE_1 = model1.input_shape[1]
SIZE_2 = model2.input_shape[1]

print("CNN models loaded successfully.")


# ====================================================
# LOAD XGBOOST
# ====================================================

xgb_model = None
scaler = None

try:
    xgb_model = joblib.load(
        "models/xgboost_dataset1.pkl"
    )

    print("XGBoost model loaded successfully.")

except Exception as e:
    print(
        f"Warning: Unable to load XGBoost model: {e}"
    )


try:
    scaler = joblib.load(
        "models/scaler_dataset1.pkl"
    )

    print("XGBoost scaler loaded successfully.")

except Exception as e:
    print(
        f"Warning: Unable to load XGBoost scaler: {e}"
    )


# ====================================================
# FEATURE EXTRACTOR
# ====================================================

print("Loading MobileNetV2 feature extractor...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

feature_extractor = base_model

print("Feature extractor loaded successfully.")


# ====================================================
# CREATE RESULTS FOLDER
# ====================================================

os.makedirs(
    "static/results",
    exist_ok=True
)


# ====================================================
# IMAGE VALIDATION
# ====================================================

mouth_cascade = None

try:

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_mcs_mouth.xml"
    )

    if os.path.exists(cascade_path):

        mouth_cascade = cv2.CascadeClassifier(
            cascade_path
        )

        if mouth_cascade.empty():
            mouth_cascade = None

except Exception:

    mouth_cascade = None


def has_oral_color(img):

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2HSV
    )

    lower_pink1 = np.array(
        [0, 40, 60]
    )

    upper_pink1 = np.array(
        [20, 255, 255]
    )

    lower_pink2 = np.array(
        [160, 30, 60]
    )

    upper_pink2 = np.array(
        [179, 255, 255]
    )

    mask1 = cv2.inRange(
        hsv,
        lower_pink1,
        upper_pink1
    )

    mask2 = cv2.inRange(
        hsv,
        lower_pink2,
        upper_pink2
    )

    mask = cv2.bitwise_or(
        mask1,
        mask2
    )

    ratio = (
        np.count_nonzero(mask)
        / mask.size
    )

    return ratio > 0.015


def has_mouth_structure(img):

    h, w = img.shape[:2]

    center = img[
        int(0.3 * h):int(0.7 * h),
        int(0.2 * w):int(0.8 * w)
    ]

    gray = cv2.cvtColor(
        center,
        cv2.COLOR_RGB2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    edge_ratio = (
        np.count_nonzero(edges)
        / edges.size
    )

    return edge_ratio > 0.03


def has_mouth_cascade(img):

    if mouth_cascade is None:
        return False

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    mouths = mouth_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(50, 50)
    )

    return len(mouths) > 0


def is_valid_image(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    # Too dark
    if np.mean(gray) < 30:
        return False

    # Too bright
    if np.mean(gray) > 220:
        return False

    # Too low contrast
    if np.var(gray) < 100:
        return False

    # Mouth cascade
    if has_mouth_cascade(img):
        return True

    # Oral color
    if has_oral_color(img):
        return True

    # Mouth structure
    if has_mouth_structure(img):
        return True

    return False


# ====================================================
# MAIN PREDICTION
# ====================================================

def predict_image(image_path):

    total_start = time.perf_counter()

    print("\n")
    print("======================================")
    print("       STARTING IMAGE ANALYSIS")
    print("======================================")


    # =================================================
    # PREPROCESSING
    # =================================================

    start = time.perf_counter()

    img_rgb, img1, img2 = preprocess(
        image_path,
        SIZE_1,
        SIZE_2
    )

    preprocess_time = (
        time.perf_counter() - start
    )

    print(
        f"[TIME] Preprocessing: "
        f"{preprocess_time:.2f}s"
    )


    if img_rgb is None:

        print("Invalid image.")

        return (
            "Invalid Image",
            None
        )


    # =================================================
    # IMAGE VALIDATION
    # =================================================

    start = time.perf_counter()

    valid = is_valid_image(
        img_rgb
    )

    validation_time = (
        time.perf_counter() - start
    )

    print(
        f"[TIME] Image validation: "
        f"{validation_time:.2f}s"
    )


    if not valid:

        print(
            "Invalid image - not an oral image."
        )

        return (
            "Invalid Image - Not an Oral Image",
            image_path
        )


    # =================================================
    # PREPARE CNN INPUT
    # =================================================

    img1 = np.expand_dims(
        img1,
        axis=0
    )

    img2 = np.expand_dims(
        img2,
        axis=0
    )


    # =================================================
    # CNN MODEL 1
    # =================================================

    print(
        "\nRunning CNN Model 1..."
    )

    start = time.perf_counter()

    prob1 = float(
        model1.predict(
            img1,
            verbose=0
        )[0][0]
    )

    model1_time = (
        time.perf_counter() - start
    )

    print(
        f"[TIME] Model 1: "
        f"{model1_time:.2f}s"
    )


    # =================================================
    # CNN MODEL 2
    # =================================================

    print(
        "Running MobileNet CNN..."
    )

    start = time.perf_counter()

    prob2 = float(
        model2.predict(
            img2,
            verbose=0
        )[0][0]
    )

    model2_time = (
        time.perf_counter() - start
    )

    print(
        f"[TIME] Model 2: "
        f"{model2_time:.2f}s"
    )


    # =================================================
    # XGBOOST
    # =================================================

    print(
        "Running XGBoost..."
    )

    start = time.perf_counter()

    if (
        xgb_model is not None
        and scaler is not None
    ):

        try:

            features = extract_xgb_features(
                image_path,
                feature_extractor,
                scaler
            )

            prob3 = float(
                xgb_model.predict_proba(
                    features
                )[0][1]
            )

        except Exception as e:

            print(
                "XGBoost prediction failed:"
            )

            print(e)

            prob3 = 0.5

    else:

        print(
            "XGBoost model unavailable."
        )

        prob3 = 0.5


    xgb_time = (
        time.perf_counter() - start
    )

    print(
        f"[TIME] XGBoost: "
        f"{xgb_time:.2f}s"
    )


    # =================================================
    # MODEL SCORES
    # =================================================

    print(
        "\n========== MODEL PREDICTIONS =========="
    )

    print(
        f"Thermal Data   : {prob1:.4f}"
    )

    print(
        f"MobileNet CNN  : {prob2:.4f}"
    )

    print(
        f"XGBoost        : {prob3:.4f}"
    )


    # =================================================
    # WEIGHTED ENSEMBLE
    # =================================================

    prob = (
        0.25 * prob1
        + 0.50 * prob2
        + 0.25 * prob3
    )

    print(
        f"Final Score    : {prob:.4f}"
    )


    # =================================================
    # DECISION
    # =================================================

    if 0.45 <= prob <= 0.55:

        result = (
            "Uncertain - Needs Medical Examination"
        )

    elif prob > 0.80:

        result = (
            "Stage 0 - No Cancer"
        )

    elif prob > 0.60:

        result = (
            "Stage 1 - Low Risk"
        )

    elif prob > 0.35:

        result = (
            "Stage 2 - Moderate Risk"
        )

    else:

        result = (
            "Stage 3 - High Risk (Likely Cancer)"
        )


    print(
        f"Prediction     : {result}"
    )


    # =================================================
    # SCORE-CAM
    # =================================================

    print(
        "\nGenerating Score-CAM..."
    )

    start = time.perf_counter()

    try:

        scorecam_map = generate_scorecam(
            img2,
            model2
        )

        scorecam_img = overlay_heatmap(
            scorecam_map,
            img_rgb
        )

        cv2.imwrite(
            "static/results/scorecam.jpg",
            cv2.cvtColor(
                scorecam_img,
                cv2.COLOR_RGB2BGR
            )
        )

        scorecam_time = (
            time.perf_counter() - start
        )

        print(
            f"[TIME] Score-CAM: "
            f"{scorecam_time:.2f}s"
        )


        # =================================================
        # LESION DETECTION
        # =================================================

        print(
            "Detecting area of interest..."
        )

        start = time.perf_counter()

        lesion_img, lesion_area, lesion_percent, bbox = detect_lesion(
            scorecam_map,
            img_rgb
        )

        lesion_time = (
            time.perf_counter() - start
        )

        print(
            f"[TIME] Lesion detection: "
            f"{lesion_time:.2f}s"
        )


        # =================================================
        # SAVE LESION IMAGE
        # =================================================

        cv2.imwrite(
            "static/results/lesion.jpg",
            cv2.cvtColor(
                lesion_img,
                cv2.COLOR_RGB2BGR
            )
        )


        # =================================================
        # REPORT
        # =================================================

        start = time.perf_counter()

        report = generate_report(
            result,
            prob1,
            prob2,
            prob3,
            lesion_area,
            lesion_percent,
            bbox,
            prob
        )

        report_time = (
            time.perf_counter() - start
        )

        print(
            f"[TIME] Report: "
            f"{report_time:.2f}s"
        )


        # =================================================
        # RESULTS
        # =================================================

        print(
            "\n========== MODEL RESULTS =========="
        )

        print(
            f"CNN Model 1      : "
            f"{prob1:.4f}"
        )

        print(
            f"MobileNet CNN    : "
            f"{prob2:.4f}"
        )

        print(
            f"XGBoost          : "
            f"{prob3:.4f}"
        )

        print(
            f"Final Ensemble   : "
            f"{prob:.4f}"
        )


        print(
            "\n========== LESION ANALYSIS =========="
        )

        print(
            f"Lesion Area      : "
            f"{lesion_area:.2f} pixels"
        )

        print(
            f"Affected Region  : "
            f"{lesion_percent:.2f}%"
        )

        print(
            f"Bounding Box     : "
            f"{bbox}"
        )


        output_path = image_path


    except Exception as e:

        print(
            "\n========== SCORE-CAM ERROR =========="
        )

        print(e)


        report = generate_report(
            result,
            prob1,
            prob2,
            prob3,
            0,
            0,
            None,
            prob
        )

        output_path = image_path


    # =================================================
    # TOTAL TIME
    # =================================================

    total_time = (
        time.perf_counter()
        - total_start
    )


    print(
        "\n======================================"
    )

    print(
        f"TOTAL ANALYSIS TIME: "
        f"{total_time:.2f}s"
    )

    print(
        "======================================"
    )

    print(
        "Analysis completed."
    )

    print(
        "======================================\n"
    )


    return (
        report,
        output_path
    )