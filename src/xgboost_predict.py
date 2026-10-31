import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def extract_xgb_features(image_path, feature_extractor, scaler):
    """
    Extract MobileNetV2 features for XGBoost prediction.
    This preprocessing matches the training notebook.
    """

    # Read image
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # Resize to MobileNet input size
    img = cv2.resize(img, (224, 224))

    # Convert BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # MobileNetV2 preprocessing
    img = preprocess_input(img.astype(np.float32))

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Extract features
    features = feature_extractor.predict(
        img,
        verbose=0
    )

    # Apply scaler used during training
    features = scaler.transform(features)

    return features