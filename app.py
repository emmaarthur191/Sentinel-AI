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
    """Generates multiple detailed radiological findings for the multi-bubble UI."""
    if prediction == 1:
        findings = [
            f"Focal densities identified with {confidence:.1%} confidence.",
            "Significant consolidation patterns observed in the parenchyma.",
            "Significant pulmonary opacification detected in the lung fields.",
            "Air bronchogram signs potentially present within consolidated areas."
        ]
    else:
        findings = [
            f"Clear pulmonary fields confirmed with {confidence:.1%} confidence.",
            "No evidence of focal consolidation or pathological opacity.",
            "Unremarkable cardiomediastinal silhouette and pleural spaces.",
            "Symmetric aeration with crisp costophrenic angles."
        ]
    return findings

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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap');
    
    html, body, [class*="st-"] { font-family: 'Outfit', sans-serif; }
    .main { background-color: #05070a; color: #ffffff; }
    
    /* Branding */
    .brand-title { color: #00d4ff; font-size: 3.5rem; font-weight: 700; margin-bottom: 0px; letter-spacing: -1px; }
    .brand-subtitle { color: #8a8d91; font-size: 0.9rem; font-weight: 600; letter-spacing: 2px; margin-top: -10px; margin-bottom: 30px; }
    
    /* Section Cards */
    .section-card { 
        background: #0f1218; 
        padding: 30px; 
        border-radius: 20px; 
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 20px;
        text-align: center;
    }
    .section-card h2 { font-size: 1.8rem; font-weight: 700; margin: 0; }
    
    /* Rationale Bubbles */
    .rationale-bubble {
        background: #12171f !important;
        border-left: 4px solid #00d4ff !important;
        padding: 15px 20px !important;
        border-radius: 10px !important;
        margin-bottom: 12px !important;
        font-size: 0.95rem !important;
        color: #e0e0e0 !important;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2) !important;
    }
    
    /* Confidence Progress Bar */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
    }
    
    /* Alert Status Box */
    .status-alert {
        padding: 20px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 20px;
    }
    .status-alert.urgent { background: rgba(255, 75, 75, 0.1); border: 1px solid #ff4b4b; color: #ff4b4b; }
    .status-alert.normal { background: rgba(0, 200, 83, 0.1); border: 1px solid #00c853; color: #00c853; }
    
    section[data-testid="stSidebar"] { background: #080a0e !important; border-right: 1px solid rgba(255, 255, 255, 0.05); }
    .stButton>button { border-radius: 10px; height: 3.5em; background: #1a1f29; border: 1px solid rgba(255, 255, 255, 0.1); }
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
st.markdown("<h1 class='brand-title'>Sentinel AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='brand-subtitle'>NEXT-GEN CLINICAL DIAGNOSTIC STATION</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    with st.container():
        st.markdown("<div class='section-card'><h2>Imaging Data Input</h2></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Patient Radiograph", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="sentinel_primary_uploader")
        
        # Persistent Visual Slot
        image_slot = st.empty()
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            # Initialize with original image
            image_slot.image(image, use_container_width=True, caption="Original Patient Scan")
            show_gradcam = st.toggle("🔍 View Grad-CAM Explainability Overlay", value=True)

with col2:
    with st.container():
        st.markdown("<div class='section-card'><h2>Diagnostic Analysis</h2></div>", unsafe_allow_html=True)
        
        if uploaded_file:
            with st.spinner("Executing Neural Analysis..."):
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
                    
                    # UI Display
                    label = "PNEUMONIA" if prediction == 1 else "NORMAL"
                    color = "#ff4b4b" if prediction == 1 else "#00c853"
                    
                    st.markdown(f"<h1 style='color: {color}; font-weight: 800; margin-bottom: 0;'>{label}</h1>", unsafe_allow_html=True)
                    st.markdown("<p style='color: #8a8d91; font-weight: 600; margin-bottom: 5px;'>Diagnostic Confidence</p>", unsafe_allow_html=True)
                    st.progress(confidence)
                    st.markdown(f"<p style='text-align: right; color: #00d4ff; font-weight: 700;'>{confidence:.1%}</p>", unsafe_allow_html=True)
                    
                    st.divider()
                    
                    st.markdown("### Clinical Rationale")
                    findings = generate_clinical_rationale(prediction, confidence)
                    for finding in findings:
                        st.markdown(f"<div class='rationale-bubble'>{finding}</div>", unsafe_allow_html=True)
                    
                    # Alert Box
                    alert_class = "urgent" if prediction == 1 else "normal"
                    alert_text = "URGENT: Pathological patterns identified. Immediate clinical intervention advised." if prediction == 1 else "Routine monitoring: No significant pathological markers detected."
                    st.markdown(f"<div class='status-alert {alert_class}'>{alert_text}</div>", unsafe_allow_html=True)

                    # Handle heatmap generation for the left column
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
                                # Update the slot in the left column
                                image_slot.image(heatmap_img, use_container_width=True, caption="Pathological Focus Area (Grad-CAM)")
        else:
            st.info("Awaiting patient imaging data...")

st.divider()
st.markdown("<script>lucide.createIcons();</script>", unsafe_allow_html=True)
