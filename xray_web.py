import base64
import html
import io
import textwrap

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

from src import gradcam

MODEL_PATH = "model/xray_model.hdf5"
CLASSES_PATH = "model/model_classes.txt"
IMG_SIZE = (180, 180)
DISPLAY_SIZE = 420
OCLUSION_SCALE = 0.5

CLASS_DISPLAY = {
    "NORMAL": ("Normal", "normal"),
    "PNEUMONIA": ("Pneumonia", "danger"),
}

st.set_page_config(
    page_title="Pneumonia Identification",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_resource
def load_grad_model():
    return gradcam.build_grad_model(load_model())


@st.cache_data(show_spinner=False)
def compute_explanations(file_bytes, class_index):
    image = Image.open(io.BytesIO(file_bytes))
    arr, _ = preprocess_image(image)
    batch = arr[np.newaxis, ...]
    heat = gradcam.compute_heatmap(load_grad_model(), batch, class_index)
    occluded = gradcam.occlusion(load_model(), batch, class_index)
    return heat, occluded


@st.cache_data
def load_class_names():
    with open(CLASSES_PATH) as f:
        return [line.strip() for line in f if line.strip()]


def preprocess_image(image_data):
    fitted = ImageOps.fit(image_data, IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(fitted.convert("RGB"), dtype=np.float32)
    gray = np.asarray(fitted.convert("L"), dtype=np.uint8)
    return arr, gray


def predict(arr, model):
    return model.predict(arr[np.newaxis, ...], verbose=0)[0]


def to_data_uri(file):
    mime = file.type or "image/jpeg"
    encoded = base64.b64encode(file.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


STYLES = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300..700;1,9..144,400..600&family=Instrument+Sans:wght@400;500;600;700&display=swap');

:root {
  --canvas: #14130f;
  --surface: #1b1a15;
  --surface-2: #232019;
  --border: #332f25;
  --text: #f3f0e8;
  --muted: #aaa598;
  --faint: #797466;
  --accent: #e7b558;
  --accent-2: #f2cd82;
  --normal: #82d3a3;
  --danger: #ef7a63;
  --radius: 22px;
  --font-sans: 'Instrument Sans', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
  --font-display: 'Fraunces', 'Iowan Old Style', Georgia, 'Times New Roman', serif;
}

html, body { background: var(--canvas); }
.stApp { background: var(--canvas); color: var(--text); font-family: var(--font-sans); }
.stApp p, .stApp li, .stApp label, .stApp input, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: var(--font-sans); }

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }

.stApp :focus-visible, [data-testid="stFileUploaderDropzone"]:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 6px; }

.stApp::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  background:
    radial-gradient(70% 45% at 50% -12%, rgba(231,181,88,.10), transparent 68%),
    radial-gradient(45% 32% at 100% 0%, rgba(130,211,163,.05), transparent 72%);
}
.block-container { position: relative; z-index: 1; padding-top: 3rem; padding-bottom: 3rem; max-width: 1120px; }

.hero { max-width: 760px; margin-bottom: .4rem; animation: rise .55s cubic-bezier(.2,.7,.2,1) both; }
.hero__eyebrow { display: inline-flex; align-items: center; gap: .55rem; font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.hero__eyebrow .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 4px rgba(231,181,88,.14); }
.hero__title { font-family: var(--font-display) !important; font-weight: 400; font-size: clamp(2.4rem, 5vw, 3.6rem); line-height: 1.02; letter-spacing: -.01em; margin: .9rem 0 .7rem; color: var(--text); }
.hero__title em { font-style: italic; color: var(--accent-2); }
.hero__sub { font-size: 1.06rem; line-height: 1.6; color: var(--muted); max-width: 56ch; margin: 0; }

.section-label { font-size: .76rem; letter-spacing: .16em; text-transform: uppercase; color: var(--faint); font-weight: 600; margin: 2.6rem 0 .7rem; animation: rise .55s .06s cubic-bezier(.2,.7,.2,1) both; }

[data-testid="stFileUploaderDropzone"] { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: .7rem; min-height: 180px; padding: 1.8rem; background: var(--surface); border: 1.5px dashed var(--border); border-radius: var(--radius); transition: border-color .22s ease, background .22s ease; cursor: pointer; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent); background: var(--surface-2); }
[data-testid="stFileUploaderDropzone"] > span { display: flex; }
[data-testid="stFileUploaderDropzone"] * { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] { background: var(--accent) !important; border: none !important; border-radius: 999px !important; font-weight: 600 !important; font-size: .95rem !important; padding: .55rem 1.3rem !important; }
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"],
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] * { color: #221a09 !important; }
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]:hover { background: var(--accent-2) !important; }
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-borderlessIcon"],
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-minimal"] { background: transparent !important; border: none !important; }
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-borderlessIcon"] svg,
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-minimal"] svg { fill: var(--muted) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { color: var(--muted) !important; font-size: .82rem !important; }
[data-testid="stFileUploaderDropzone"] .stFileChip { background: var(--surface-2) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
[data-testid="stFileUploaderDropzone"] .stFileChip svg { fill: var(--muted) !important; }
[data-testid="stFileChipName"] { color: var(--text) !important; }

.empty { border: 1px solid var(--border); border-radius: var(--radius); padding: 1.8rem; display: grid; grid-template-columns: auto 1fr; gap: 1.1rem; align-items: center; margin-top: 1.1rem; background: linear-gradient(180deg, var(--surface), rgba(27,26,21,.45)); animation: rise .5s cubic-bezier(.2,.7,.2,1) both; }
.empty__icon { width: 46px; height: 46px; border-radius: 14px; background: rgba(231,181,88,.12); display: grid; place-items: center; color: var(--accent); }
.empty__title { font-weight: 600; font-size: 1rem; color: var(--text); margin: 0 0 .25rem; }
.empty__text { color: var(--muted); font-size: .92rem; line-height: 1.55; margin: 0; }

.result { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.05fr); gap: 1.4rem; margin-top: 1.1rem; align-items: stretch; }
.result > * { animation: rise .5s cubic-bezier(.2,.7,.2,1) both; }
.result .verdict { animation-delay: .09s; }

.lightbox { margin: 0; background: #0b0a08; border: 1px solid var(--border); border-radius: var(--radius); padding: 14px; position: relative; overflow: hidden; display: flex; flex-direction: column; }
.lightbox::after { content: ""; position: absolute; inset: 0; pointer-events: none; background: radial-gradient(60% 45% at 50% 42%, rgba(231,181,88,.07), transparent 70%); }
.lightbox img { width: 100%; max-height: 440px; object-fit: contain; border-radius: 12px; display: block; margin: auto; }
.lightbox figcaption { margin-top: .7rem; font-size: .74rem; letter-spacing: .06em; color: var(--faint); text-transform: uppercase; font-weight: 600; word-break: break-all; }

.verdict { border: 1px solid var(--border); border-radius: var(--radius); padding: 1.7rem 1.7rem 1.5rem; background: linear-gradient(180deg, var(--surface-2), var(--surface)); position: relative; overflow: hidden; display: flex; flex-direction: column; }
.verdict::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.verdict--danger::before { background: linear-gradient(90deg, var(--danger), transparent); }
.verdict--normal::before { background: linear-gradient(90deg, var(--normal), transparent); }
.verdict__top { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
.verdict__eyebrow { font-size: .74rem; letter-spacing: .16em; text-transform: uppercase; color: var(--faint); font-weight: 600; }
.chip { font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; padding: .3rem .7rem; border-radius: 999px; white-space: nowrap; }
.chip--danger { color: var(--danger); background: rgba(239,122,99,.13); border: 1px solid rgba(239,122,99,.3); }
.chip--normal { color: var(--normal); background: rgba(130,211,163,.13); border: 1px solid rgba(130,211,163,.3); }
.verdict__label { font-family: var(--font-display) !important; font-weight: 500; font-size: clamp(2rem, 4vw, 2.9rem); line-height: 1.05; margin: .85rem 0 .1rem; color: var(--text); }
.verdict__conf { display: flex; align-items: baseline; gap: .45rem; margin: .1rem 0 1.4rem; }
.verdict__conf .num { font-size: 2.1rem; font-weight: 700; color: var(--text); font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
.verdict__conf .unit { font-size: 1.05rem; font-weight: 600; color: var(--muted); }
.verdict__conf .cap { font-size: .82rem; color: var(--faint); margin-left: .3rem; }
.bars { display: flex; flex-direction: column; gap: .95rem; margin-top: auto; }
.bar__row { display: flex; justify-content: space-between; font-size: .82rem; margin-bottom: .4rem; }
.bar__name { font-weight: 600; letter-spacing: .04em; text-transform: uppercase; color: var(--faint); }
.bar__val { font-variant-numeric: tabular-nums; color: var(--text); font-weight: 600; }
.bar__track { height: 8px; border-radius: 999px; background: rgba(255,255,255,.06); overflow: hidden; }
.bar__fill { height: 100%; border-radius: 999px; animation: grow 1s cubic-bezier(.2,.8,.2,1) both; }
.bar__fill--danger { background: linear-gradient(90deg, #ef7a63, #f2a18d); }
.bar__fill--normal { background: linear-gradient(90deg, #82d3a3, #a9e5c1); }
.verdict__note { font-size: .78rem; line-height: 1.5; color: var(--faint); margin: 1.4rem 0 0; border-top: 1px solid var(--border); padding-top: .95rem; }

.foot { margin-top: 3rem; padding-top: 1.3rem; border-top: 1px solid var(--border); display: flex; flex-wrap: wrap; gap: .5rem 1.4rem; justify-content: space-between; color: var(--faint); font-size: .78rem; }
.foot span { display: inline-flex; align-items: center; gap: .45rem; }
.foot .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--faint); }

.panels { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin-top: 1.1rem; }
.panel { margin: 0; background: #0b0a08; border: 1px solid var(--border); border-radius: 18px; padding: 12px; display: flex; flex-direction: column; gap: .65rem; animation: rise .5s cubic-bezier(.2,.7,.2,1) both; }
.panel:nth-child(2) { animation-delay: .08s; }
.panel:nth-child(3) { animation-delay: .16s; }
.panel img { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 12px; display: block; }
.panel figcaption { font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; color: var(--faint); font-weight: 600; text-align: center; }
.legend { display: flex; align-items: center; gap: .85rem; margin-top: 1rem; flex-wrap: wrap; }
.legend__grad { flex: 1 1 220px; height: 10px; border-radius: 999px; background: linear-gradient(90deg, #2b4dff, #2bd6ff, #3ce07a, #f2d24b, #ef5a3c); }
.legend__cap { font-size: .78rem; color: var(--muted); }
.legend__cap--low { color: #8fb0ff; }
.legend__cap--high { color: var(--danger); }
.explain-note { font-size: .82rem; line-height: 1.6; color: var(--muted); margin: .95rem 0 0; }
.explain-note strong { color: var(--text); font-weight: 600; }

[data-testid="stSpinner"] { color: var(--accent); }
[data-testid="stSpinner"] svg, [data-testid="stSpinner"] i { color: var(--accent); }

@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes grow { from { width: 0; } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
@media (max-width: 860px) { .result { grid-template-columns: 1fr; } .lightbox img { max-height: 360px; } .panels { grid-template-columns: 1fr; } }
@media (max-width: 520px) { .empty { grid-template-columns: 1fr; } .block-container { padding-left: 1rem; padding-right: 1rem; } }
"""

HERO = textwrap.dedent("""
    <header class="hero">
      <div class="hero__eyebrow"><span class="dot"></span> Chest radiograph analysis</div>
      <h1 class="hero__title">Pneumonia <em>identification</em></h1>
      <p class="hero__sub">Upload a chest X-ray and a trained convolutional model classifies it as normal or pneumonia, with the confidence behind the call.</p>
    </header>
""").strip()

EMPTY_STATE = textwrap.dedent("""
    <div class="empty">
      <div class="empty__icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
      </div>
      <div>
        <p class="empty__title">No radiograph yet</p>
        <p class="empty__text">Drop a JPEG or PNG chest X-ray above to run the model. The scan and its classification will appear here.</p>
      </div>
    </div>
""").strip()

FOOTER = textwrap.dedent("""
    <footer class="foot">
      <span><span class="dot"></span> Educational tool &middot; not a clinical diagnosis</span>
      <span>Custom CNN &middot; Grad-CAM &amp; occlusion &middot; 180&times;180 input</span>
    </footer>
""").strip()


def render_result(file, probs, class_names):
    idx = int(np.argmax(probs))
    label = class_names[idx]
    display, tone = CLASS_DISPLAY.get(label, (label.title(), "normal"))
    confidence = 100 * float(probs[idx])
    tag = "Flagged" if tone == "danger" else "Clear"

    bars = ""
    for name, prob in sorted(zip(class_names, probs), key=lambda pair: -pair[1]):
        _, bar_tone = CLASS_DISPLAY.get(name, (name, "normal"))
        bars += (
            '<div class="bar">'
            f'<div class="bar__row"><span class="bar__name">{html.escape(name)}</span>'
            f'<span class="bar__val">{100 * float(prob):.1f}%</span></div>'
            f'<div class="bar__track"><div class="bar__fill bar__fill--{bar_tone}" '
            f'style="width:{100 * float(prob):.2f}%"></div></div>'
            "</div>"
        )

    return (
        '<div class="result">'
        '<figure class="lightbox">'
        f'<img src="{to_data_uri(file)}" alt="Uploaded chest radiograph">'
        f"<figcaption>{html.escape(file.name)}</figcaption>"
        "</figure>"
        f'<section class="verdict verdict--{tone}">'
        '<div class="verdict__top">'
        '<div class="verdict__eyebrow">Screening result</div>'
        f'<div class="chip chip--{tone}">{tag}</div>'
        "</div>"
        f'<h2 class="verdict__label">{html.escape(display)}</h2>'
        '<div class="verdict__conf">'
        f'<span class="num">{confidence:.1f}</span><span class="unit">%</span>'
        '<span class="cap">model confidence</span>'
        "</div>"
        f'<div class="bars">{bars}</div>'
        '<p class="verdict__note">Classification from the trained model. Review by a qualified clinician is required '
        "before any diagnostic decision.</p>"
        "</section>"
        "</div>"
    )


def build_panel_images(gray, display, size, alpha, blur=0.0):
    gray_disp = gradcam.resize_uint8(gray, (size, size))
    scaled = gradcam.resize_uint8((np.clip(display, 0.0, 1.0) * 255).astype(np.uint8), (size, size))
    if blur:
        scaled = gradcam.blur_uint8(scaled, blur)
    base_rgb = gradcam.gray_to_rgb(gray_disp)
    heat_rgb = gradcam.jet(scaled / 255.0)
    overlay_rgb = gradcam.overlay_intensity(base_rgb, heat_rgb, scaled / 255.0, alpha)
    return base_rgb, overlay_rgb, heat_rgb


def render_explainability(base_rgb, cam_rgb, occ_rgb, alpha, label):
    display, _ = CLASS_DISPLAY.get(label, (label.title(), "normal"))
    panels = [
        (gradcam.to_data_uri(base_rgb), "Radiograph"),
        (gradcam.to_data_uri(cam_rgb), f"Grad-CAM &middot; {alpha:.0%}"),
        (gradcam.to_data_uri(occ_rgb), f"Region importance &middot; {alpha:.0%}"),
    ]
    figures = "".join(
        f'<figure class="panel"><img src="{uri}" alt="{caption}">'
        f"<figcaption>{caption}</figcaption></figure>"
        for uri, caption in panels
    )
    return (
        f'<div class="panels">{figures}</div>'
        '<div class="legend">'
        '<span class="legend__cap legend__cap--low">Low influence</span>'
        '<span class="legend__grad"></span>'
        '<span class="legend__cap legend__cap--high">High influence</span>'
        "</div>"
        f'<p class="explain-note">Warm colours (yellow, red) mark regions the model leans on for '
        f'<strong>{html.escape(display)}</strong>; blue marks regions that barely matter. '
        "<strong>Grad-CAM</strong> reads the gradient of the output; on this model it under-reports pneumonia. "
        "<strong>Region importance</strong> (occlusion) masks each area and measures how much the prediction "
        "drops, so it is the more reliable of the two. Neither is proof of disease.</p>"
    )


st.markdown(f"<style>{STYLES}</style>", unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)

with st.spinner("Preparing the model..."):
    model = load_model()

class_names = load_class_names()

st.markdown('<div class="section-label">Radiograph</div>', unsafe_allow_html=True)

file = st.file_uploader(
    "Upload a chest X-ray",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if file is None:
    st.markdown(EMPTY_STATE, unsafe_allow_html=True)
else:
    try:
        image = Image.open(file)
        image.load()
    except Exception:
        st.error("That file could not be read as an image. Please upload a JPEG or PNG chest X-ray.")
    else:
        with st.spinner("Analyzing radiograph..."):
            arr, gray = preprocess_image(image)
            probabilities = predict(arr, model)
        st.markdown(render_result(file, probabilities, class_names), unsafe_allow_html=True)

        st.markdown('<div class="section-label">Explainability &middot; Grad-CAM &amp; occlusion</div>', unsafe_allow_html=True)
        alpha = st.slider("Heatmap opacity", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        predicted_index = int(np.argmax(probabilities))
        try:
            with st.spinner("Mapping model attention..."):
                heat, occluded = compute_explanations(file.getvalue(), predicted_index)
            cam_display = gradcam.to_display(heat)
            occ_display = gradcam.scale_to_unit(occluded, OCLUSION_SCALE)
        except Exception:
            st.info("The model attention map could not be generated for this image.")
        else:
            if float(cam_display.max()) <= 0.0 and float(occ_display.max()) <= 0.0:
                st.info("The model produced no localised attention for this image, so no heatmap is shown.")
            else:
                base_rgb, cam_rgb, _ = build_panel_images(gray, cam_display, DISPLAY_SIZE, alpha)
                _, occ_rgb, _ = build_panel_images(gray, occ_display, DISPLAY_SIZE, alpha, blur=8)
                st.markdown(
                    render_explainability(base_rgb, cam_rgb, occ_rgb, alpha, class_names[predicted_index]),
                    unsafe_allow_html=True,
                )

st.markdown(FOOTER, unsafe_allow_html=True)
