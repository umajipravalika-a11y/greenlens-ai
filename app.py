from pathlib import Path
import time
import random

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Optional dependency: app still starts if Ultralytics is not installed.
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

APP_DIR = Path(__file__).resolve().parent
CUSTOM_MODEL_PATH = APP_DIR / "best.pt"
DEFAULT_MODEL_PATH = APP_DIR / "yolov8n.pt"

st.set_page_config(
    page_title="GreenLens AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------ Theme ---------------------------------------
st.markdown(
    """
    <style>
    :root { --green: #10b981; --green-dark: #047857; --slate: #0f172a; }
    .stApp { background: linear-gradient(135deg, #f8fafc 0%, #ecfdf5 100%); }
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0; }
    .hero {
      padding: 1.5rem 1.7rem; border-radius: 20px; color: white;
      background: linear-gradient(120deg, #064e3b, #0f766e 60%, #0f172a);
      border: 1px solid rgba(16,185,129,.4); margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: clamp(1.7rem, 3vw, 2.5rem); }
    .hero p { margin: .55rem 0 0 0; color: #d1fae5; }
    .metric-card {
      background: white; border: 1px solid #d1fae5; border-radius: 16px;
      padding: 1rem; min-height: 115px; box-shadow: 0 4px 18px rgba(15,23,42,.04);
    }
    .metric-title { color: #64748b; font-size: .84rem; }
    .metric-value { color: #065f46; font-weight: 800; font-size: 1.35rem; margin-top: .2rem; }
    .metric-note { color: #475569; font-size: .78rem; margin-top: .25rem; }
    .rule-card {
      background: white; border: 1px solid #dbeafe; border-left: 5px solid #10b981;
      border-radius: 12px; padding: 1rem; margin: .65rem 0; color: #0f172a;
    }
    .tag { display:inline-block; background:#d1fae5; color:#065f46; padding:.2rem .55rem;
      border-radius:99px; font-size:.75rem; font-weight:700; }
    .muted { color: #64748b; font-size: .9rem; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #d1fae5;
      padding: .75rem; border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------- Guidance rule engine ----------------------------
RULES = {
    "battery": {
        "title": "Battery",
        "hazard": "High fire hazard",
        "severity": "HIGH",
        "protocol": (
            "Do not crush or dispose of in standard trash bins. "
            "Keep away from heat and isolate from loose metal objects. "
            "Take it to a designated municipal e-waste or battery collection point."
        ),
        "icon": "🔋",
        "category": "E-waste",
    },
    "lithium battery": {
        "title": "Lithium Battery",
        "hazard": "High fire hazard",
        "severity": "HIGH",
        "protocol": (
            "Do not crush, puncture, or place in standard trash. "
            "Keep away from heat and loose metal objects, and use a designated "
            "battery collection point. If swollen, leaking, or hot, do not handle; "
            "contact local waste-safety services."
        ),
        "icon": "🔋",
        "category": "E-waste",
    },
    "circuit board": {
        "title": "Circuit Board / PCB",
        "hazard": "Potential toxic heavy metals",
        "severity": "HIGH",
        "protocol": (
            "Do not dismantle or burn. Circuit boards can contain valuable metals "
            "and hazardous substances. Hand over to a certified e-waste recycler."
        ),
        "icon": "🧩",
        "category": "E-waste",
    },
    "pcb": {
        "title": "Circuit Board / PCB",
        "hazard": "Potential toxic heavy metals",
        "severity": "HIGH",
        "protocol": (
            "Do not dismantle or burn. Hand over to a certified e-waste recycler."
        ),
        "icon": "🧩",
        "category": "E-waste",
    },
    "plastic bottle": {
        "title": "Plastic Bottle",
        "hazard": "Non-biodegradable waste",
        "severity": "MEDIUM",
        "protocol": (
            "Empty, rinse, and flatten if local rules allow. Place in the designated "
            "plastic recycling stream; follow local recycling guidance for caps and labels."
        ),
        "icon": "🧴",
        "category": "Recyclable",
    },
    "bottle": {
        "title": "Bottle / Recyclable",
        "hazard": "May become persistent litter",
        "severity": "MEDIUM",
        "protocol": (
            "Empty and rinse the container, then use the appropriate local recycling bin. "
            "Check the material and local collection rules."
        ),
        "icon": "♻️",
        "category": "Recyclable",
    },
    "cell phone": {
        "title": "Electronic Device",
        "hazard": "Contains battery and electronic components",
        "severity": "HIGH",
        "protocol": (
            "Do not put electronics in household trash. Use an authorized e-waste "
            "collection or take-back program."
        ),
        "icon": "📱",
        "category": "E-waste",
    },
    "laptop": {
        "title": "Electronic Device",
        "hazard": "Contains battery and electronic components",
        "severity": "HIGH",
        "protocol": (
            "Use an authorized e-waste collection or take-back program. "
            "If possible, remove personal data before recycling."
        ),
        "icon": "💻",
        "category": "E-waste",
    },
}

def guidance_for(label: str) -> dict:
    normalized = label.strip().lower().replace("_", " ").replace("-", " ")
    if normalized in RULES:
        return RULES[normalized]
    if "battery" in normalized:
        return RULES["battery"]
    if "circuit" in normalized or normalized == "pcb":
        return RULES["circuit board"]
    if "bottle" in normalized or "plastic" in normalized:
        return RULES["plastic bottle"]
    if normalized in {"cell phone", "mobile phone", "laptop", "computer"}:
        return RULES.get(normalized, RULES["cell phone"])
    return {
        "title": label.title(),
        "hazard": "Check material before disposal",
        "severity": "INFO",
        "protocol": (
            "Confirm the item's material and follow your local municipal waste rules. "
            "For electronic equipment, use an authorized e-waste collection point."
        ),
        "icon": "♻️",
        "category": "Review",
    }

# -------------------------- Image utilities --------------------------------
def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def draw_detections(image_bgr: np.ndarray, detections: list) -> np.ndarray:
    output = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = str(det["label"])
        conf = float(det["confidence"])
        color = (16, 185, 129) if guidance_for(label)["severity"] != "HIGH" else (0, 165, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        caption = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        top = max(0, y1 - th - 12)
        cv2.rectangle(output, (x1, top), (x1 + tw + 10, top + th + 10), color, -1)
        cv2.putText(
            output, caption, (x1 + 5, top + th + 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA
        )
    return output

def load_model():
    """Prefer custom weights, then try YOLOv8 Nano; cache in session state."""
    if "model" in st.session_state:
        return st.session_state["model"], st.session_state.get("model_name", "")
    if YOLO is None:
        st.session_state["model"] = None
        st.session_state["model_name"] = ""
        return None, ""
    model_path = CUSTOM_MODEL_PATH if CUSTOM_MODEL_PATH.exists() else DEFAULT_MODEL_PATH
    try:
        model = YOLO(str(model_path))
        st.session_state["model"] = model
        st.session_state["model_name"] = model_path.name
        return model, model_path.name
    except Exception as exc:
        st.session_state["model"] = None
        st.session_state["model_name"] = ""
        st.session_state["model_error"] = str(exc)
        return None, ""

def infer_yolo(image_bgr: np.ndarray, confidence: float, image_size: int):
    model, model_name = load_model()
    if model is None:
        return [], None, "unavailable", ""
    results = model.predict(
        source=bgr_to_rgb(image_bgr),
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )
    detections = []
    if results:
        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "label": str(names.get(cls_id, cls_id)),
                    "confidence": conf,
                    "box": [x1, y1, x2, y2],
                })
    return detections, results, "model", model_name

def demo_detections(image_bgr: np.ndarray):
    """Illustrative boxes only; not an AI prediction."""
    h, w = image_bgr.shape[:2]
    # Stable illustrative labels based on image dimensions, not image content.
    options = [
        ("Plastic Bottle", 0.94, (0.12, 0.20, 0.47, 0.84)),
        ("Battery", 0.88, (0.55, 0.43, 0.83, 0.72)),
    ]
    chosen = options[:1] if w < 500 else options
    return [
        {
            "label": label,
            "confidence": confidence,
            "box": [x1 * w, y1 * h, x2 * w, y2 * h],
        }
        for label, confidence, (x1, y1, x2, y2) in chosen
    ]

# ------------------------------- Sidebar ------------------------------------
with st.sidebar:
    st.markdown("## 🌿 GreenLens AI")
    st.caption("Computer vision for responsible waste sorting")
    st.divider()
    st.markdown("### ⚙️ Detection settings")
    confidence = st.slider("Confidence threshold", 0.10, 0.95, 0.35, 0.05)
    image_size = st.select_slider("Inference image size", options=[320, 416, 512, 640], value=640)
    use_demo = st.toggle("Use illustrative demo detections", value=False)
    st.markdown("### 🧠 Model status")
    if CUSTOM_MODEL_PATH.exists():
        st.success("Custom model found: best.pt")
    elif DEFAULT_MODEL_PATH.exists():
        st.info("Using local yolov8n.pt")
    else:
        st.warning("No local weights found. YOLO may download yolov8n.pt on first run.")
    st.caption("For best results, train best.pt on your own battery, PCB, and recyclable dataset.")
    st.divider()
    st.markdown("### ♻️ Sorting reminder")
    st.write("Never crush batteries or burn/dismantle electronic waste. Follow local collection rules.")

# -------------------------------- Header ------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🌿 GreenLens AI</h1>
      <p><b>Real-Time Urban E-Waste &amp; Recyclable Sorting Assistant</b></p>
      <p>Vignan Institute of Technology &amp; Science · CSE (Data Science)</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write(
    "Capture or upload an item to inspect it, review detected classes, and get practical "
    "disposal guidance. Model results depend on the labels and objects represented in the weights."
)

# ------------------------------ Input + analysis ----------------------------
left, right = st.columns([0.95, 1.15], gap="large")
uploaded_image = None
with left:
    st.subheader("📷 1. Capture or upload")
    input_method = st.radio("Choose input", ["Upload image", "Camera"], horizontal=True)
    if input_method == "Upload image":
        uploaded = st.file_uploader(
            "Choose a JPG, JPEG, or PNG image",
            type=["jpg", "jpeg", "png"],
            help="Use a clear image with the item visible.",
        )
        if uploaded is not None:
            try:
                uploaded_image = Image.open(uploaded).convert("RGB")
            except Exception:
                st.error("That file could not be read as an image.")
    else:
        camera_file = st.camera_input("Take a picture of the item")
        if camera_file is not None:
            try:
                uploaded_image = Image.open(camera_file).convert("RGB")
            except Exception:
                st.error("The camera image could not be read.")

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Input image", use_container_width=True)
    else:
        st.info("Upload an image or capture one with your camera to begin.")

with right:
    st.subheader("🔎 2. Analysis & rule engine")
    if uploaded_image is None:
        st.markdown(
            """
            <div class="rule-card">
              <span class="tag">READY</span>
              <h4>Waiting for an image</h4>
              <p>Detected objects and disposal rules will appear here after analysis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        image_bgr = pil_to_bgr(uploaded_image)
        if st.button("🌱 Analyze item", type="primary", use_container_width=True):
            with st.spinner("Inspecting image…"):
                start = time.perf_counter()
                if use_demo:
                    detections = demo_detections(image_bgr)
                    mode, model_name = "demo", "Illustrative demo"
                else:
                    try:
                        detections, _, mode, model_name = infer_yolo(image_bgr, confidence, image_size)
                    except Exception as exc:
                        detections, mode, model_name = [], "error", ""
                        st.session_state["inference_error"] = str(exc)
                    if mode == "unavailable":
                        detections = demo_detections(image_bgr)
                        mode, model_name = "demo", "Illustrative demo (model unavailable)"
                elapsed_ms = (time.perf_counter() - start) * 1000
                st.session_state["analysis_result"] = {
                    "detections": detections,
                    "mode": mode,
                    "model_name": model_name,
                    "elapsed_ms": elapsed_ms,
                    "source_image": image_bgr,
                }

        result = st.session_state.get("analysis_result")
        if result is not None:
            detections = result["detections"]
            st.image(
                bgr_to_rgb(draw_detections(result["source_image"], detections)),
                caption="Annotated image",
                use_container_width=True,
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Objects found", len(detections))
            m2.metric("Analysis time", f'{result["elapsed_ms"]:.0f} ms')
            m3.metric("Mode", "Demo" if result["mode"] == "demo" else "YOLO")
            if result["mode"] == "demo":
                st.warning(
                    "Demo mode: boxes and labels are illustrative placeholders, not predictions "
                    "from the image. Install model dependencies and provide valid weights for real detection."
                )
            elif result["mode"] == "error":
                st.error("Inference failed. Check the model weights and installed dependencies.")
                if st.session_state.get("inference_error"):
                    st.caption(st.session_state["inference_error"])
            else:
                st.caption(f'Model used: {result["model_name"] or "YOLO"}')

            if not detections:
                st.info(
                    "No objects passed the confidence threshold. Try a clearer image, lower the "
                    "threshold, or use a model trained for these object classes."
                )
            else:
                st.markdown("#### Detected items & handling guidance")
                for index, det in enumerate(detections, start=1):
                    rule = guidance_for(det["label"])
                    with st.container(border=True):
                        st.markdown(
                            f'{rule["icon"]} **{rule["title"]}**  '
                            f'`{det["confidence"]:.0%} confidence`'
                        )
                        st.markdown(
                            f'**Hazard:** {rule["hazard"]} · **Priority:** {rule["severity"]}'
                        )
                        st.write(rule["protocol"])
                        st.caption(f'Category: {rule["category"]} · Detection #{index}')

# ---------------------------- Social impact ---------------------------------
st.divider()
st.subheader("🌍 Designed for measurable environmental impact")
st.caption(
    "These are product goals and alignment themes, not independently verified performance claims."
)
metric_cols = st.columns(5)
metrics = [
    ("Zero-cost core", "Open-source stack", "No paid inference API required"),
    ("Fast inference", "Target: sub-second", "Depends on device, model, and image size"),
    ("Safer handling", "Hazard awareness", "Prompts safer e-waste routing"),
    ("UN SDG 11", "Sustainable cities", "Responsible urban waste systems"),
    ("UN SDGs 12 & 13", "Responsible action", "Consumption, production & climate awareness"),
]
for col, (title, value, note) in zip(metric_cols, metrics):
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">{title}</div>'
            f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
            unsafe_allow_html=True,
        )

st.divider()
with st.expander("ℹ️ Model setup, limitations & responsible use"):
    st.markdown(
        """
        - **Custom weights:** Put `best.pt` beside `app.py`. It should be a compatible Ultralytics YOLO model
          trained with classes that match your waste items.
        - **Pretrained fallback:** If `best.pt` is absent, GreenLens tries `yolov8n.pt`. Standard COCO weights
          recognize common objects such as bottles, but generally do not include dedicated battery or PCB classes.
        - **First run:** Ultralytics may download `yolov8n.pt`; internet access may be required.
        - **Demo mode:** Use the sidebar toggle to preview the interface without model weights. Demo boxes are
          illustrative and must never be treated as actual detections.
        - **Privacy:** Uploaded and camera images are processed by this app. If you deploy it publicly, review
          hosting, logging, retention, and privacy settings before use.
        - **Safety:** Visual detection is not a safety certification. For damaged, swollen, leaking, or hot batteries,
          avoid handling and consult local waste-safety services.
        """
    )

st.caption("GreenLens AI · Prototype for educational and environmental-awareness use")
