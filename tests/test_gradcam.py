import numpy as np
import pytest
import tensorflow as tf

from src import gradcam


def tiny_model():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu", input_shape=(16, 16, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )


def test_last_conv_layer_finds_conv2d():
    assert gradcam.last_conv_layer(tiny_model()).name.startswith("conv2d")


def test_last_conv_layer_raises_without_conv():
    dense_only = tf.keras.Sequential([tf.keras.layers.Dense(3, input_shape=(5,))])
    with pytest.raises(ValueError):
        gradcam.last_conv_layer(dense_only)


def test_build_grad_model_outputs_activations_and_predictions():
    grad_model = gradcam.build_grad_model(tiny_model())
    conv_out, preds = grad_model(np.zeros((1, 16, 16, 1), dtype=np.float32))
    assert tuple(conv_out.shape) == (1, 16, 16, 4)
    assert tuple(preds.shape) == (1, 2)


def test_compute_heatmap_shape_and_range():
    grad_model = gradcam.build_grad_model(tiny_model())
    batch = np.random.default_rng(0).random((1, 16, 16, 1)).astype(np.float32) * 255.0
    heat = gradcam.compute_heatmap(grad_model, batch, class_index=1)
    assert heat.shape == (16, 16)
    assert heat.min() >= 0.0
    assert heat.max() <= 1.0
    # Grad-CAM returns an all-zero map when no gradient flows; otherwise it is
    # normalised so the peak activation is exactly 1.
    if heat.any():
        assert heat.max() == pytest.approx(1.0)


def test_jet_ramp_endpoints_and_shape():
    low = gradcam.jet(np.array(0.0))
    high = gradcam.jet(np.array(1.0))
    assert low.tolist() == [0, 0, 255]
    assert high.tolist() == [255, 0, 0]
    ramp = gradcam.jet(np.linspace(0, 1, 11))
    assert ramp.shape == (11, 3)
    assert ramp.dtype == np.uint8


def test_jet_clips_out_of_range_values():
    assert gradcam.jet(np.array(-3.0)).tolist() == [0, 0, 255]
    assert gradcam.jet(np.array(9.0)).tolist() == [255, 0, 0]


def test_gray_to_rgb_replicates_channels():
    gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    rgb = gradcam.gray_to_rgb(gray)
    assert rgb.shape == (2, 2, 3)
    assert np.array_equal(rgb[..., 0], gray)
    assert np.array_equal(rgb[..., 1], gray)
    assert np.array_equal(rgb[..., 2], gray)


def test_resize_uint8_resamples_to_width_height():
    out = gradcam.resize_uint8(np.zeros((10, 10), dtype=np.uint8), (20, 8))
    assert out.shape == (8, 20)
    assert out.dtype == np.uint8


def test_overlay_alpha_endpoints():
    base = np.full((4, 4, 3), 40, dtype=np.uint8)
    heat = np.full((4, 4, 3), 200, dtype=np.uint8)
    assert np.array_equal(gradcam.overlay(base, heat, 0.0), base)
    assert np.array_equal(gradcam.overlay(base, heat, 1.0), heat)
    mid = gradcam.overlay(base, heat, 0.5)
    assert mid.mean() == pytest.approx(120, abs=1)


def test_to_display_clips_to_percentile_and_lifts_midtones():
    heat = np.full((10, 10), 0.05, dtype=np.float32)
    heat[0, 0] = 1.0
    out = gradcam.to_display(heat)
    assert out.max() == pytest.approx(1.0)
    assert out.min() >= 0.0
    # a mid-tone pixel must come out brighter than its raw value, not crushed to 0
    assert out[5, 5] > 0.05


def test_to_display_all_zero_map_stays_zero():
    out = gradcam.to_display(np.zeros((5, 5), dtype=np.float32))
    assert np.all(out == 0)


def test_overlay_intensity_alpha_endpoints():
    base = np.full((4, 4, 3), 40, dtype=np.uint8)
    heat = np.full((4, 4, 3), 200, dtype=np.uint8)
    intensity = np.ones((4, 4), dtype=np.float64)
    assert np.array_equal(gradcam.overlay_intensity(base, heat, intensity, 0.0), base)
    assert np.array_equal(gradcam.overlay_intensity(base, heat, intensity, 1.0), heat)


def test_overlay_intensity_keeps_low_areas_closer_to_base():
    base = np.full((4, 4, 3), 40, dtype=np.uint8)
    heat = np.full((4, 4, 3), 200, dtype=np.uint8)
    low = gradcam.overlay_intensity(base, heat, np.zeros((4, 4)), 0.6)
    high = gradcam.overlay_intensity(base, heat, np.ones((4, 4)), 0.6)
    assert low.mean() < high.mean()


def test_occlusion_returns_input_shaped_nonnegative_map():
    batch = np.random.default_rng(1).random((1, 16, 16, 1)).astype(np.float32) * 255.0
    heat = gradcam.occlusion(tiny_model(), batch, class_index=1, patch=6, stride=3)
    assert heat.shape == (16, 16)
    assert heat.min() >= 0.0


def test_occlusion_marks_a_region_that_drives_the_score():
    # a model whose score depends only on the top-left 8x8 block
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(16, 16, 1)),
            tf.keras.layers.Cropping2D(((0, 8), (0, 8))),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, use_bias=False, kernel_initializer="ones"),
        ]
    )
    model.layers[-1].set_weights([np.ones((64, 1), dtype=np.float32)])
    batch = np.ones((1, 16, 16, 1), dtype=np.float32) * 100.0
    heat = gradcam.occlusion(model, batch, class_index=0, patch=8, stride=8)
    assert heat[:8, :8].mean() > heat[8:, 8:].mean()


def test_scale_to_unit_uses_absolute_ceiling():
    assert gradcam.scale_to_unit(np.array([0.5], dtype=np.float32), 0.5)[0] == pytest.approx(1.0)
    assert gradcam.scale_to_unit(np.array([0.05], dtype=np.float32), 0.5)[0] < 0.5
    assert np.all(gradcam.scale_to_unit(np.zeros((2, 2), dtype=np.float32), 0.5) == 0)


def test_blur_uint8_spreads_a_bright_pixel():
    img = np.zeros((21, 21), dtype=np.uint8)
    img[10, 10] = 255
    out = gradcam.blur_uint8(img, 1.5)
    assert out.shape == (21, 21)
    assert out.dtype == np.uint8
    assert out[10, 10] < 255
    assert out[10, 11] > 0


def test_occlusion_shape_and_non_negative():
    model = tiny_model()
    batch = np.random.default_rng(1).random((1, 16, 16, 1)).astype(np.float32) * 255.0
    heat = gradcam.occlusion(model, batch, class_index=1, patch=8, stride=8)
    assert heat.shape == (16, 16)
    assert heat.min() >= 0.0


def test_scale_to_unit_is_absolute_and_bounded():
    out = gradcam.scale_to_unit(np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float32), ceiling=0.5)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out[2] == pytest.approx(1.0)
    assert out[1] < out[2]
    assert np.all(gradcam.scale_to_unit(np.zeros(4, dtype=np.float32), ceiling=0.5) == 0)


def test_blur_uint8_preserves_shape_and_dtype():
    out = gradcam.blur_uint8(np.zeros((12, 12), dtype=np.uint8), 3)
    assert out.shape == (12, 12)
    assert out.dtype == np.uint8


def test_to_data_uri_is_png_base64():
    uri = gradcam.to_data_uri(np.zeros((4, 4, 3), dtype=np.uint8))
    assert uri.startswith("data:image/png;base64,")
