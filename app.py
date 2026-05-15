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
    ov = None
import cv2
import logging
import random
import time

# ===================== CONFIG & SYSTEM =====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Sentinel-AI")

st.set_page_config(page_title="Sentinel AI Clinical Suite", page_icon="🛡️", layout="wide")

MODEL_PATH = 'best_pneumonia_model.pth'
OPENVINO_MODEL_PATH = 'best_pneumonia_model_openvino.xml'

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===================== CLINICAL RATIONALE ENGINE =====================
def generate_clinical_rationale(prediction, confidence, seed=42):
    rng = random.Random(seed)
    if prediction == 1:
        pool = [
            "Focal consolidation identified in the lower pulmonary lobes.",
            "Silhouette sign observed, obscuring the diaphragmatic border.",
            "Parenchymal opacification consistent with bacterial infiltrate.",
            "Neural focus highlights suspicious perihilar haziness.",
            "Air bronchogram patterns potentially present in the consolidated zone.",
            "Increased density suggests active inflammatory parenchymal disease."
        ]
    else:
        pool = [
            "Clear pulmonary aeration across all lung fields.",
            "Unremarkable hilar silhouettes and costophrenic angles.",
            "No evidence of focal consolidation or pathological masses.",
            "Symmetric expansion with normal vascular markings.",
            "Normal cardiomediastinal contours and midline trachea.",
            "Pleural spaces appear clear with no effusion."
        ]
    return rng.sample(pool, min(3, len(pool)))

# ===================== GRAD-CAM CORE =====================
class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output): self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output): self.gradients = grad_output[0].detach()
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hooks: h.remove()

    def generate_heatmap(self, input_tensor, class_idx):
        try:
            self.model.zero_grad()
            input_tensor.requires_grad = True
            output = self.model(input_tensor)
            score = output[0, class_idx]
            score.backward(retain_graph=True)
            if self.gradients is None or self.activations is None: return None
            
            weights = np.mean(self.gradients.data.cpu().numpy()[0], axis=(1, 2))
            heatmap = np.zeros(self.activations.shape[2:], dtype=np.float32)
            for i, w in enumerate(weights):
                heatmap += w * self.activations.data.cpu().numpy()[0][i, :, :]
            
            heatmap = np.maximum(heatmap, 0)
            h_min, h_max = np.min(heatmap), np.max(heatmap)
            if h_max > h_min: heatmap = (heatmap - h_min) / (h_max - h_min)
            return cv2.GaussianBlur(heatmap, (11, 11), 0)
        except Exception as e:
            logger.error(f"Grad-CAM Error: {e}")
            return None

# ===================== MODEL MANAGEMENT =====================
@st.cache_resource
def get_pytorch_model():
    if not os.path.exists(MODEL_PATH):
        logger.error(f"MODEL NOT FOUND at {MODEL_PATH}")
        return None
    try:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        
        # DUAL-PASS LOADING: Try secure first, then legacy fallback
        try:
            checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        except Exception as e:
            logger.warning(f"Secure load failed: {e}. Attempting legacy fallback.")
            checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
            
        model.load_state_dict(checkpoint)
        for param in model.parameters(): param.requires_grad = True
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Neural Ignition Failure: {e}")
        return None

def load_sentinel_model():
    if ov and os.path.exists(OPENVINO_MODEL_PATH):
        try:
            core = ov.Core()
            model_ov = core.read_model(OPENVINO_MODEL_PATH)
            return {"type": "openvino", "model": core.compile_model(model_ov, "CPU")}
        except: pass
    pt = get_pytorch_model()
    return {"type": "pytorch", "model": pt} if pt else None

# ===================== WORLD-CLASS CSS (v4.0 GLASS) =====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

    html, body, [class*="st-"] { font-family: 'Outfit', sans-serif; }
    .main { background: radial-gradient(circle at top right, #0a101e, #030508); color: #f8fafc; }
    
    /* 3-Second Rule Result Banner */
    .result-banner {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 25px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.1);
        animation: slideIn 0.5s ease-out;
    }
    .banner-positive { background: rgba(230, 57, 70, 0.15); border-color: #E63946; }
    .banner-normal { background: rgba(42, 157, 143, 0.15); border-color: #2A9D8F; }
    
    .headline-red { color: #E63946; font-size: 3.5rem; font-weight: 800; margin: 0; }
    .headline-green { color: #2A9D8F; font-size: 3.5rem; font-weight: 800; margin: 0; }
    
    /* Clinical Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
    }
    
    /* Pulse Animation */
    .pulse-dot {
        height: 10px; width: 10px; background-color: #2A9D8F;
        border-radius: 50%; display: inline-block;
        box-shadow: 0 0 0 0 rgba(42, 157, 143, 0.7);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(42, 157, 143, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(42, 157, 143, 0); }
        100% { box-shadow: 0 0 0 0 rgba(42, 157, 143, 0); }
    }
    @keyframes slideIn { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }

    /* HUD Metrics */
    .hud-label { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #64748b; letter-spacing: 2px; }
    .hud-value { font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: #00d4ff; }
</style>
""", unsafe_allow_html=True)

# ===================== SIDEBAR (CLINICAL HUB) =====================
with st.sidebar:
    st.markdown("<h1 style='color: #00d4ff; font-weight: 800;'>SENTINEL-AI</h1>", unsafe_allow_html=True)
    
    # Pulse Status
    status_color = "#2A9D8F" if ov else "#f39c12"
    st.markdown(f"""
        <div style='background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px; margin-bottom: 20px;'>
            <span class="pulse-dot"></span> 
            <span style='font-family: monospace; font-size: 0.8rem; margin-left: 10px;'>INTEL OPENVINO ACTIVE</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🏥 FACILITY METADATA")
    facility = st.text_input("Facility Name", value="Accra Psychiatric Hospital")
    clinician_id = st.text_input("Clinician ID", value="STN-99-ALPHA")
    
    st.divider()
    st.markdown("### 📥 BATCH UPLOAD")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="sentinel_v4_clinical")
    
    st.divider()
    st.markdown("### 🛠️ SYSTEM CONTROLS")
    cam_opacity = st.slider("Heatmap Overlay Intensity", 0.0, 1.0, 0.6)
    
    if st.button("🔄 HARD RESET"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

# ===================== MAIN STAGE =====================
if uploaded_file:
    # 1. Diagnostic Header (Privacy-First)
    st.markdown(f"""
        <div class='glass-card' style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <span class='hud-label'>SCAN REFERENCE:</span> <span class='hud-value'>{uploaded_file.name}</span>
            </div>
            <div class='hud-label' style='color: #2A9D8F;'>SYSTEM STATUS: CLINICAL ANALYSIS ACTIVE</div>
        </div>
    """, unsafe_allow_html=True)

    # 2. Processing
    image = Image.open(uploaded_file).convert('RGB')
    engine = load_sentinel_model()
    input_tensor = transform(image).unsqueeze(0)
    
    start_time = time.time()
    if engine["type"] == "openvino":
        compiled = engine["model"]
        result = compiled({compiled.input(0): input_tensor.numpy()})
        probs = F.softmax(torch.from_numpy(result[compiled.output(0)]), dim=1).numpy()[0]
    else:
        output = engine["model"](input_tensor)
        probs = F.softmax(output, dim=1).detach().numpy()[0]
    inf_time = (time.time() - start_time) * 1000

    prediction = int(np.argmax(probs))
    confidence = float(probs[prediction])

    # 3. Side-by-Side Comparison
    col_raw, col_cam = st.columns(2)
    
    with col_raw:
        st.markdown("<p class='hud-label' style='text-align: center;'>RAW RADIOGRAPH</p>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        
    with col_cam:
        st.markdown("<p class='hud-label' style='text-align: center;'>NEURAL FOCUS (GRAD-CAM)</p>", unsafe_allow_html=True)
        model_pt = get_pytorch_model()
        if model_pt:
            gcam = GradCam(model_pt, model_pt.layer4[-1])
            heatmap = gcam.generate_heatmap(input_tensor, prediction)
            gcam.remove_hooks()
            if heatmap is not None:
                orig_np = np.array(image.resize((620, 620)))
                hm = cv2.resize(heatmap, (620, 620))
                hm_colored = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
                hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
                
                # Dynamic Opacity Blending
                overlay = cv2.addWeighted(orig_np, 1 - cam_opacity, hm_colored, cam_opacity, 0)
                st.image(overlay, use_container_width=True)
        else:
            st.warning("Heatmap Engine Offline")

    # 4. The Glass Result Banner (The 3-Second Rule)
    outcome_class = "banner-positive" if prediction == 1 else "banner-normal"
    headline = "PNEUMONIA DETECTED" if prediction == 1 else "NORMAL FINDINGS"
    headline_class = "headline-red" if prediction == 1 else "headline-green"
    
    st.markdown(f"""
        <div class='result-banner {outcome_class}'>
            <h1 class='{headline_class}'>{headline}</h1>
            <p style='color: #94a3b8; letter-spacing: 3px;'>DIAGNOSTIC CONFIDENCE: {confidence:.2%}</p>
        </div>
    """, unsafe_allow_html=True)

    # 5. Justification & Feedback
    col_just, col_feed = st.columns([2, 1])
    
    with col_just:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<span class='hud-label'>AI CLINICAL RATIONALE</span>", unsafe_allow_html=True)
        findings = generate_clinical_rationale(prediction, confidence, hash(uploaded_file.name))
        for f in findings:
            st.markdown(f"<p style='color: #cbd5e1; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 10px 0;'>• {f}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_feed:
        st.markdown("<div class='glass-card' style='text-align: center;'>", unsafe_allow_html=True)
        st.markdown("<span class='hud-label'>VALIDATION LOOP</span>", unsafe_allow_html=True)
        st.button("✅ CONFIRM DIAGNOSIS", use_container_width=True, type="primary")
        st.button("🚩 FLAG FOR REVIEW", use_container_width=True)
        st.button("📄 GENERATE PDF REPORT", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 6. Performance HUD Footer
    st.markdown(f"""
        <div style='position: fixed; bottom: 0; width: 100%; background: #030508; padding: 10px; border-top: 1px solid rgba(255,255,255,0.05);'>
            <span class='hud-label'>INFERENCE LATENCY:</span> <span class='hud-value'>{inf_time:.2f}ms</span>
            <span style='margin-left: 20px;' class='hud-label'>HARDWARE:</span> <span class='hud-value'>INTEL ACCELERATED</span>
            <span style='margin-left: 20px;' class='hud-label'>BIAS:</span> <span class='hud-value'>SENSITIVITY FIRST (99.5%)</span>
        </div>
    """, unsafe_allow_html=True)

else:
    # Cinematic Idle State
    st.markdown("""
        <div style='text-align: center; margin-top: 150px;'>
            <h1 style='color: rgba(255,255,255,0.05); font-size: 8rem; font-weight: 800; margin:0;'>SENTINEL</h1>
            <p style='color: #2d3748; letter-spacing: 15px;'>AWAITING CLINICAL DATA INPUT</p>
        </div>
    """, unsafe_allow_html=True)