"""Grad-CAM explainability for the pneumonia classifier.

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the pixels the
network leaned on most when it made its prediction. Warm colours (yellow, red)
mark the regions that pushed the prediction toward the chosen class; cool
colours (blue) mark regions that contributed little.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter

# Anchors of the "jet" colour ramp: blue -> cyan -> green -> yellow -> red.
_JET_ANCHORS = np.array(
    [
        [0, 0, 255],
        [0, 255, 255],
        [0, 255, 0],
        [255, 255, 0],
        [255, 0, 0],
    ],
    dtype=np.float64,
)
_JET_POSITIONS = np.linspace(0.0, 1.0, len(_JET_ANCHORS))


def last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Return the final Conv2D layer, which Grad-CAM uses for its activations."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("Model has no Conv2D layer to explain.")


def build_grad_model(model: tf.keras.Model) -> tf.keras.Model:
    """Build a model returning (last-conv activations, predictions).

    Models loaded from HDF5 often have no functional graph, so ``layer.output``
    is undefined. Re-chaining the layers through a fresh ``Input`` restores it.
    """
    target = last_conv_layer(model)
    inputs = tf.keras.Input(shape=model.input_shape[1:])
    activations = inputs
    conv_output = None
    for layer in model.layers:
        activations = layer(activations)
        if layer is target:
            conv_output = activations
    return tf.keras.Model(inputs, [conv_output, activations])


def compute_heatmap(grad_model: tf.keras.Model, batch: np.ndarray, class_index: int) -> np.ndarray:
    """Compute a Grad-CAM heatmap in [0, 1] for one image and one class.

    ``batch`` is a single preprocessed image with a leading batch axis, and the
    values must match the model's expected input contract (raw 0-255 here).
    """
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(batch)
        class_score = predictions[:, class_index]
    gradients = tape.gradient(class_score, conv_output)

    pooled = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heat = tf.reduce_sum(conv_output[0] * pooled, axis=-1)
    heat = tf.nn.relu(heat).numpy()

    peak = float(heat.max())
    if peak > 0:
        heat = heat / peak
    return heat


def jet(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to the jet colour ramp as uint8 RGB.

    Low values become blue, high values become red - matching "cold = little
    attention, hot = strong attention".
    """
    x = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    channels = [np.interp(x, _JET_POSITIONS, _JET_ANCHORS[:, c]) for c in range(3)]
    return np.stack(channels, axis=-1).astype(np.uint8)


def occlusion(model: tf.keras.Model, batch: np.ndarray, class_index: int, patch: int = 30, stride: int = 15, baseline: float = 0.0) -> np.ndarray:
    """Perturbation (occlusion) sensitivity as a replacement for Grad-CAM.

    Each patch is masked in turn and the drop in the target class probability is
    recorded. Unlike Grad-CAM this needs no gradients, so it stays meaningful for
    this model where the pneumonia map's gradients cancel out. All perturbations
    are evaluated in a single batched pass.
    """
    array = np.asarray(batch)[0]
    height, width = array.shape[:2]
    rows = list(range(0, height - patch + 1, stride))
    cols = list(range(0, width - patch + 1, stride))
    if rows[-1] != height - patch:
        rows.append(height - patch)
    if cols[-1] != width - patch:
        cols.append(width - patch)

    variants = []
    for r in rows:
        for c in cols:
            occluded = array.copy()
            occluded[r:r + patch, c:c + patch] = baseline
            variants.append(occluded)

    base = float(model.predict(np.asarray(batch), verbose=0)[0][class_index])
    scores = model.predict(np.stack(variants), verbose=0)[:, class_index].reshape(len(rows), len(cols))
    drops = np.clip(base - scores, 0.0, None)

    heat = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            heat[r:r + patch, c:c + patch] += drops[i, j]
            counts[r:r + patch, c:c + patch] += 1.0
    return heat / np.maximum(counts, 1.0)


def to_display(heatmap: np.ndarray, upper_percentile: float = 99.0, gamma: float = 0.6) -> np.ndarray:
    """Stretch a raw CAM into a displayable [0, 1] map.

    Raw Grad-CAM maps are spiky: ReLU zeroes most pixels and a single peak holds
    the maximum, so dividing by the maximum crushes everything else to ~0 - which
    ``jet`` renders as solid blue. Clipping to an upper percentile and applying a
    gamma below 1 restores the mid-tones so the map actually reads.
    """
    heat = np.asarray(heatmap, dtype=np.float32)
    ceiling = float(np.percentile(heat, upper_percentile))
    if ceiling <= 0:
        return np.zeros_like(heat)
    return np.clip(heat / ceiling, 0.0, 1.0) ** gamma


def scale_to_unit(values: np.ndarray, ceiling: float, gamma: float = 0.6) -> np.ndarray:
    """Map values onto [0, 1] against an absolute ceiling (not per-image max).

    Used for occlusion drops, where the magnitude is meaningful: a chest with a
    large drop stays hot, a diffuse one stays cool, instead of both being
    stretched to look equally intense.
    """
    v = np.asarray(values, dtype=np.float32)
    if ceiling <= 0:
        return np.zeros_like(v)
    return np.clip(v / ceiling, 0.0, 1.0) ** gamma


def blur_uint8(image: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian-smooth a 2-D uint8 map (occlusion grids are coarse and blocky)."""
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="L")
    return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius)), dtype=np.uint8)


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    """Replicate a single-channel uint8 image to three channels."""
    gray = np.asarray(gray, dtype=np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def resize_uint8(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resample a 2-D uint8 array to ``size`` as (width, height)."""
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="L")
    return np.asarray(pil.resize(size, Image.Resampling.BILINEAR), dtype=np.uint8)


def overlay(base_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float) -> np.ndarray:
    """Blend a heatmap over a base image; alpha 0 keeps the base, 1 the heatmap."""
    a = float(np.clip(alpha, 0.0, 1.0))
    base = np.asarray(base_rgb, dtype=np.float64)
    heat = np.asarray(heat_rgb, dtype=np.float64)
    return ((1.0 - a) * base + a * heat).astype(np.uint8)


def overlay_intensity(base_rgb: np.ndarray, heat_rgb: np.ndarray, intensity: np.ndarray, alpha: float, floor: float = 0.35) -> np.ndarray:
    """Blend a heatmap with an alpha that rises with the activation.

    A constant-alpha blend tints the whole radiograph blue. Making the opacity
    follow the normalised activation keeps low-attention areas a faint blue wash
    while high-attention areas glow strongly.
    """
    weight = alpha * (floor + (1.0 - floor) * np.asarray(intensity, dtype=np.float64))
    a = np.clip(weight, 0.0, 1.0)[..., np.newaxis]
    base = np.asarray(base_rgb, dtype=np.float64)
    heat = np.asarray(heat_rgb, dtype=np.float64)
    return np.clip(base * (1.0 - a) + heat * a, 0.0, 255.0).astype(np.uint8)


def to_data_uri(image_rgb: np.ndarray) -> str:
    """Encode a uint8 RGB array as a base64 PNG data URI for embedding in HTML."""
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
