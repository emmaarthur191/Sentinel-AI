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

# --- SYSTEM CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Sentinel-AI")

# --- DYNAMIC SECURITY ALLOWLIST ---
# This ensures compatibility across different NumPy/PyTorch versions
safe_globals = [
    torch._utils._rebuild_tensor_v2,
    torch.storage.TypedStorage,
    torch.FloatStorage,
    'collections.OrderedDict'
]

# Add NumPy internal functions if they exist in the current environment
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
OPENVINO_BIN_PATH = 'best_pneumonia_model_openvino.bin'

# --- IMAGE PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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
            # Ensure gradients are enabled for this specific pass
            with torch.enable_grad():
                input_tensor.requires_grad = True
                output = self.model(input_tensor)
                self.model.zero_grad()
                loss = output[0, class_idx]
                loss.backward()
                
                if self.gradients is None or self.activations is None:
                    logger.error("Hooks failed to capture gradients/activations")
                    return None

                gradients = self.gradients.data.cpu().numpy()
                activations = self.activations.data.cpu().numpy()
                
                weights = np.mean(gradients, axis=(2, 3))[0]
                heatmap = np.zeros(activations.shape[2:], dtype=np.float32)

                for i, w in enumerate(weights):
                    heatmap += w * activations[0, i, :, :]

                heatmap = np.maximum(heatmap, 0)
                heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1
                return heatmap
        except Exception as e:
            logger.error(f"Heatmap generation failed: {e}")
            return None

# --- MODEL LOADING LOGIC ---
# --- MODEL LOADING LOGIC ---
@st.cache_resource
def get_pytorch_model():
    """Load and cache the native PyTorch model for Grad-CAM."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
        # Security-hardened loading
        try:
            checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint)
        except Exception:
            checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint)
            
        model.eval()
        return model
    except Exception as e:
        logger.error(f"PyTorch Load Failure: {e}")
        return None

@st.cache_resource
def get_openvino_model():
    """Load and cache the OpenVINO model for high-speed inference."""
    if not (ov and os.path.exists(OPENVINO_MODEL_PATH)):
        return None
    try:
        core = ov.Core()
        model_ov = core.read_model(OPENVINO_MODEL_PATH)
        compiled_model = core.compile_model(model_ov, "CPU")
        return compiled_model
    except Exception as e:
        logger.warning(f"OpenVINO Engine Failure: {e}")
        return None

def load_sentinel_model():
    """Hybrid loader that prioritizes OpenVINO but ensures PyTorch is ready."""
    ov_model = get_openvino_model()
    if ov_model:
        return {"type": "openvino", "model": ov_model}
    
    pt_model = get_pytorch_model()
    if pt_model:
        return {"type": "pytorch", "model": pt_model}
    
    return None

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Sentinel AI | Clinical Diagnostics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE ---
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

def reset_station():
    st.session_state.uploader_key += 1
    st.rerun()

# --- STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #262730; color: white; border: 1px solid #4a4a4a; }
    .stButton>button:hover { border-color: #ff4b4b; color: #ff4b4b; }
    .report-card { background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b; margin-top: 10px; }
    .metric-box { text-align: center; padding: 10px; background: rgba(255, 255, 255, 0.03); border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Sentinel AI")
    st.caption("Clinical Diagnostic Station v1.2")
    st.divider()
    
    st.subheader("Patient Vitals")
    age = st.number_input("Patient Age", min_value=0, max_value=120, value=25, key="input_age")
    spo2 = st.number_input("SPO2 (%)", min_value=0, max_value=100, value=98, step=1, key="input_spo2")
    temp = st.number_input("Temp (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1, key="input_temp")
    
    if st.button("🔄 New Patient Scan", use_container_width=True, key="btn_new_scan", on_click=reset_station):
        pass

# --- MAIN UI ---
st.title("Clinical Chest Radiograph Analysis")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📥 Imaging Data Input")
    uploaded_file = st.file_uploader("Select Chest Radiograph (JPG/PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key=f"file_radiograph_{st.session_state.uploader_key}")
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True, caption="Original Radiograph")
        
        show_gradcam = st.toggle("🔍 View Grad-CAM Explainability Overlay", value=False, key="toggle_gradcam")

with col2:
    st.markdown("### 🧬 AI Diagnostic Output")
    
    if uploaded_file:
        with st.spinner("Processing High-Resolution Inference..."):
            engine_data = load_sentinel_model()
            
            if not engine_data:
                st.error("Diagnostic Engine Offline. Please check model files.")
            else:
                # Preprocess
                input_tensor = transform(image).unsqueeze(0)
                
                # Inference
                if engine_data["type"] == "openvino":
                    results = engine_data["model"]([input_tensor.numpy()])[0]
                    probs = F.softmax(torch.from_numpy(results), dim=1).numpy()[0]
                else:
                    with torch.no_grad():
                        output = engine_data["model"](input_tensor)
                        probs = F.softmax(output, dim=1).numpy()[0]
                
                prediction = int(np.argmax(probs))
                confidence = float(probs[prediction])
                
                # Grad-CAM if requested
                heatmap_img = None
                if show_gradcam:
                    model_pt = get_pytorch_model()
                    if model_pt:
                        gcam = GradCam(model_pt, model_pt.layer4[-1])
                        try:
                            heatmap = gcam.generate_heatmap(input_tensor, prediction)
                            gcam.remove_hooks()
                            
                            if heatmap is not None:
                                # Apply heatmap to image
                                original_np = np.array(image.resize((224, 224)))
                                heatmap_resized = cv2.resize(heatmap, (224, 224))
                                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                                overlay = cv2.addWeighted(original_np, 0.6, heatmap_colored, 0.4, 0)
                                heatmap_img = Image.fromarray(overlay)
                        except Exception as e:
                            st.warning(f"Grad-CAM generation error: {e}")

                # UI Display
                label = "PNEUMONIA" if prediction == 1 else "NORMAL"
                color = "#ff4b4b" if prediction == 1 else "#00c853"
                
                st.markdown(f"<h2 style='color: {color}; text-align: center;'>{label} DETECTED</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>Confidence: {confidence:.2%}</p>", unsafe_allow_html=True)
                
                if heatmap_img:
                    st.image(heatmap_img, use_container_width=True, caption="Pathological Focus Area (Grad-CAM)")
                
                # Report Card
                st.markdown(f"""
                <div class="report-card">
                    <h4>Clinical Findings</h4>
                    <p><b>Observation:</b> {"Increased pulmonary opacification and consolidation consistent with infectious process." if prediction == 1 else "Clear lung fields with no visible consolidations or significant opacities."}</p>
                    <p><b>Recommendation:</b> {"Stat radiological consultation and clinical correlation suggested." if prediction == 1 else "Routine monitoring."}</p>
                    <p style="font-size: 0.8em; opacity: 0.7;">Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Upload a patient scan to begin clinical analysis.")

st.divider()
st.caption("Sentinel AI is a diagnostic aid and should be used with clinical judgment.")
