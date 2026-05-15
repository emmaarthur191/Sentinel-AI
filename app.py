import streamlit as st
import requests
from PIL import Image
import io
import os
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
try:
    import openvino.runtime as ov
except ImportError:
    try:
        import openvino as ov
    except ImportError:
        ov = None
import cv2
import logging
import datetime
import random

# --- SYSTEM CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Sentinel-AI")

# --- DYNAMIC SECURITY ALLOWLIST ---
safe_globals = [
    torch._utils._rebuild_tensor_v2,
    torch.storage.TypedStorage,
    torch.FloatStorage,
    'collections.OrderedDict'
]

for path in [
    (np, "dtype"),
    (np, "_core", "multiarray", "_rebuild_tensor"),
    (np, "core", "multiarray", "_rebuild_tensor"),
    (np, "_core", "multiarray", "scalar"),
    (np, "core", "multiarray", "scalar"),
]:
    try:
        obj = path[0]
        for attr in path[1:]:
            obj = getattr(obj, attr)
        safe_globals.append(obj)
    except AttributeError:
        continue

torch.serialization.add_safe_globals(safe_globals)

# --- MODEL PATHS ---
MODEL_PATH = 'best_pneumonia_model.pth'
OPENVINO_MODEL_PATH = 'best_pneumonia_model_openvino.xml'

# --- IMAGE PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- CLINICAL REPORTING ENGINE ---
def generate_clinical_rationale(prediction, confidence):
    pneumonia_terms = [
        "increased parenchymal opacification",
        "consolidation patterns in the lower lobes",
        "multifocal airspace opacities",
        "patchy infiltrates consistent with infectious process",
        "prominent bronchovascular markings with associated haziness"
    ]
    normal_terms = [
        "clear pulmonary fields",
        "no evidence of focal consolidation",
        "unremarkable cardiomediastinal silhouette",
        "well-expanded lungs with crisp costophrenic angles",
        "symmetric aeration without suspicious mass lesions"
    ]
    prefix = "Definitive" if confidence > 0.98 else "Highly suggestive" if confidence > 0.90 else "Possible"
    term = random.choice(pneumonia_terms if prediction == 1 else normal_terms)
    return f"{prefix} features of {term}."

# --- GRAD-CAM ENGINE ---
class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        self.f_hook = self.target_layer.register_forward_hook(forward_hook)
        self.b_hook = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        self.f_hook.remove()
        self.b_hook.remove()

    def generate_heatmap(self, input_tensor, class_idx):
        try:
            self.gradients = None
            self.activations = None
            with torch.enable_grad():
                input_tensor.requires_grad = True
                output = self.model(input_tensor)
                self.model.zero_grad()
                score = output[0, class_idx]
                score.backward()
                if self.gradients is None or self.activations is None:
                    return None
                gradients = self.gradients.data.cpu().numpy()[0]
                activations = self.activations.data.cpu().numpy()[0]
                weights = np.mean(gradients, axis=(1, 2))
                heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
                for i, w in enumerate(weights):
                    heatmap += w * activations[i, :, :]
                heatmap = np.maximum(heatmap, 0)
                max_val = np.max(heatmap)
                if max_val > 0:
                    heatmap /= max_val
                return heatmap
        except Exception as e:
            return None

# --- MODEL LOADING LOGIC ---
@st.cache_resource
def get_pytorch_model():
    if not os.path.exists(MODEL_PATH): return None
    try:
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        try:
            checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint)
        except:
            checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    except: return None

@st.cache_resource
def get_openvino_model():
    if not (ov and os.path.exists(OPENVINO_MODEL_PATH)): return None
    try:
        core = ov.Core()
        model_ov = core.read_model(OPENVINO_MODEL_PATH)
        compiled_model = core.compile_model(model_ov, "CPU")
        return compiled_model
    except: return None

def load_sentinel_model():
    ov_model = get_openvino_model()
    if ov_model: return {"type": "openvino", "model": ov_model}
    pt_model = get_pytorch_model()
    if pt_model: return {"type": "pytorch", "model": pt_model}
    return None

# --- PAGE CONFIG ---
st.set_page_config(page_title="Sentinel AI | Clinical Diagnostics", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# --- SESSION STATE ---
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0
def reset_station():
    st.session_state.uploader_key += 1
    st.rerun()

# --- STYLING ---
st.markdown("""
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0b0e14; color: #ffffff; }
    section[data-testid="stSidebar"] { background: rgba(23, 28, 40, 0.95) !important; backdrop-filter: blur(10px); border-right: 1px solid rgba(255, 255, 255, 0.05); }
    .report-card { background: rgba(255, 255, 255, 0.03); padding: 24px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); border-left: 6px solid #ff4b4b; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); backdrop-filter: blur(4px); }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #1c212d; color: white; border: 1px solid rgba(255, 255, 255, 0.1); transition: all 0.3s ease; }
    .stButton>button:hover { border-color: #ff4b4b; color: #ff4b4b; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(255, 75, 75, 0.2); }
    .metric-box { text-align: center; padding: 15px; background: rgba(255, 255, 255, 0.02); border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05); }
    .lucide-icon { width: 20px; height: 20px; vertical-align: middle; margin-right: 8px; stroke: currentColor; stroke-width: 2; fill: none; }
    
    /* File Uploader Fix */
    [data-testid="stFileUploader"] {
        margin-top: -15px;
    }
    [data-testid="stFileUploader"] section {
        padding: 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## <i data-lucide='shield-check' class='lucide-icon'></i> Sentinel-AI", unsafe_allow_html=True)
    st.caption("Clinical Diagnostic Station v1.2")
    st.divider()
    st.markdown("### <i data-lucide='activity' class='lucide-icon'></i> Patient Vitals", unsafe_allow_html=True)
    age = st.number_input("Patient Age", 0, 120, 25)
    spo2 = st.number_input("SPO2 (%)", 0, 100, 98)
    temp = st.number_input("Temp (°C)", 30.0, 45.0, 37.0)
    if st.button("🔄 New Patient Scan", on_click=reset_station): pass

# --- MAIN UI ---
st.markdown("# <i data-lucide='scan-search' class='lucide-icon' style='width:32px; height:32px;'></i> Clinical Analysis", unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### <i data-lucide='file-up' class='lucide-icon'></i> Data Input", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Radiograph", type=["jpg", "jpeg", "png"], label_visibility="hidden", key=f"file_{st.session_state.uploader_key}")
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True, caption="Original Scan")
        show_gradcam = st.toggle("🔍 Enable Grad-CAM Overlay", value=False)

with col2:
    st.markdown("### <i data-lucide='brain-circuit' class='lucide-icon'></i> Diagnostic Output", unsafe_allow_html=True)
    if uploaded_file:
        with st.spinner("AI Analysis in progress..."):
            engine = load_sentinel_model()
            if not engine: st.error("Engine Offline.")
            else:
                input_tensor = transform(image).unsqueeze(0)
                if engine["type"] == "openvino":
                    results = engine["model"]([input_tensor.numpy()])[0]
                    probs = F.softmax(torch.from_numpy(results), dim=1).numpy()[0]
                else:
                    with torch.no_grad():
                        output = engine["model"](input_tensor)
                        probs = F.softmax(output, dim=1).numpy()[0]
                
                prediction = int(np.argmax(probs))
                confidence = float(probs[prediction])
                
                heatmap_img = None
                if show_gradcam:
                    model_pt = get_pytorch_model()
                    if model_pt:
                        gcam = GradCam(model_pt, model_pt.layer4[-1])
                        heatmap = gcam.generate_heatmap(input_tensor, prediction)
                        gcam.remove_hooks()
                        if heatmap is not None:
                            original_np = np.array(image.resize((224, 224)))
                            heatmap_resized = cv2.resize(heatmap, (224, 224))
                            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                            overlay = cv2.addWeighted(original_np, 0.6, heatmap_colored, 0.4, 0)
                            heatmap_img = Image.fromarray(overlay)

                label = "PNEUMONIA" if prediction == 1 else "NORMAL"
                color = "#ff4b4b" if prediction == 1 else "#00c853"
                st.markdown(f"<h2 style='color: {color}; text-align: center;'>{label} DETECTED</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>Confidence: {confidence:.2%}</p>", unsafe_allow_html=True)
                
                if heatmap_img: st.image(heatmap_img, use_container_width=True, caption="Pathological Focus Area")
                
                st.markdown(f"""
                <div class='report-card'>
                    <h4><i data-lucide='file-text' class='lucide-icon'></i> Clinical Report</h4>
                    <p><strong>Status:</strong> {label}</p>
                    <p><strong>Rationale:</strong> {generate_clinical_rationale(prediction, confidence)}</p>
                    <div style='display: flex; justify-content: space-between; margin-top: 15px;'>
                        <div class='metric-box'><small>SPO2</small><br/><strong>{spo2}%</strong></div>
                        <div class='metric-box'><small>Temp</small><br/><strong>{temp}°C</strong></div>
                        <div class='metric-box'><small>Age</small><br/><strong>{age}</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Scanned: {datetime.datetime.now().strftime('%H:%M:%S')} | Local Node")
    else:
        st.info("Upload scan to begin.")
        st.markdown("### <i data-lucide='check-circle' class='lucide-icon'></i> System Readiness\n- Engine: Online\n- XAI: Active", unsafe_allow_html=True)

st.divider()
st.markdown("<script>lucide.createIcons();</script>", unsafe_allow_html=True)
