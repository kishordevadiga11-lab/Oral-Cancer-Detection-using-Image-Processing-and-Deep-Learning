import cv2
import numpy as np
import tensorflow as tf


def generate_scorecam(image, model, max_N=16):
    """
    Faster Score-CAM implementation.

    Instead of calling model.predict() once for every
    feature map, all masked images are created first
    and evaluated in a single batch.

    max_N is reduced from 32 to 16 to improve speed while
    still retaining the strongest feature maps.
    """

    # ====================================================
    # GET LAST CONVOLUTIONAL LAYER
    # ====================================================

    last_conv_layer = model.get_layer("Conv_1")

    activation_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=last_conv_layer.output
    )

    # ====================================================
    # GET FEATURE MAPS
    # ====================================================

    conv_outputs = activation_model(
        image,
        training=False
    )

    conv_outputs = conv_outputs[0].numpy()

    H, W, C = conv_outputs.shape

    # ====================================================
    # SELECT STRONGEST FEATURE MAPS
    # ====================================================

    importance = np.mean(
        np.maximum(conv_outputs, 0),
        axis=(0, 1)
    )

    # Use only the strongest feature maps
    N = min(max_N, C)

    top_idx = np.argsort(
        importance
    )[::-1][:N]

    # ====================================================
    # CREATE ALL MASKED IMAGES
    # ====================================================

    original = image[0].numpy() if tf.is_tensor(image) else image[0]

    masked_images = []

    for idx in top_idx:

        fmap = conv_outputs[:, :, idx]

        # Resize feature map to input image size
        fmap = cv2.resize(
            fmap,
            (image.shape[2], image.shape[1]),
            interpolation=cv2.INTER_LINEAR
        )

        # Keep only positive activations
        fmap = np.maximum(fmap, 0)

        # Normalize feature map
        max_value = fmap.max()

        if max_value > 1e-8:
            fmap = fmap / max_value
        else:
            fmap = np.zeros_like(fmap)

        # Create masked image
        masked = original * fmap[:, :, np.newaxis]

        masked_images.append(masked)

    # ====================================================
    # BATCH SCORE PREDICTION
    # ====================================================

    if len(masked_images) == 0:
        return np.zeros(
            (H, W),
            dtype=np.float32
        )

    masked_batch = np.asarray(
        masked_images,
        dtype=np.float32
    )

    # IMPORTANT:
    # One model call instead of N model calls
    predictions = model.predict(
        masked_batch,
        batch_size=N,
        verbose=0
    )

    predictions = np.asarray(
        predictions
    ).reshape(-1)

    # ====================================================
    # GENERATE WEIGHTED HEATMAP
    # ====================================================

    heatmap = np.zeros(
        (H, W),
        dtype=np.float32
    )

    for weight, idx in zip(
        predictions,
        top_idx
    ):

        fmap = conv_outputs[:, :, idx]

        heatmap += (
            float(weight) * fmap
        )

    # ====================================================
    # NORMALIZE HEATMAP
    # ====================================================

    heatmap = np.maximum(
        heatmap,
        0
    )

    max_heatmap = heatmap.max()

    if max_heatmap > 1e-8:

        heatmap /= max_heatmap

    else:

        heatmap = np.zeros_like(
            heatmap
        )

    return heatmap.astype(
        np.float32
    )


def overlay_heatmap(heatmap, image):

    # ====================================================
    # RESIZE HEATMAP
    # ====================================================

    heatmap = cv2.resize(
        heatmap,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    # ====================================================
    # IMPROVE VISUALIZATION
    # ====================================================

    heatmap = np.power(
        heatmap,
        0.6
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    # ====================================================
    # APPLY COLOR MAP
    # ====================================================

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # ====================================================
    # OVERLAY
    # ====================================================

    output = cv2.addWeighted(
        image,
        0.65,
        heatmap,
        0.35,
        0
    )

    return output