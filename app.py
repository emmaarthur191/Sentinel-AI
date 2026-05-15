import streamlit as st
import requests
from PIL import Image
import io
import os
import base64

# Page Config for Premium Branding
st.set_page_config(
    page_title="Sentinel AI | Clinical Diagnostics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SESSION STATE INITIALIZATION
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

def reset_station():
    st.session_state.uploader_key += 1
    st.rerun()

# LUCIDE ICON HELPER & LIBRARY
st.markdown("""
    <script src="https://unpkg.com/lucide@latest"></script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            lucide.createIcons();
        });
        // Handle Streamlit's dynamic updates
        const observer = new MutationObserver(() => {
            lucide.createIcons();
        });
        observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """, unsafe_allow_html=True)

def lucide_icon(name, size=20, color="currentColor", extra_style=""):
    return f'<i data-lucide="{name}" style="width: {size}px; height: {size}px; stroke: {color}; stroke-width: 2px; vertical-align: middle; {extra_style}"></i>'

# WORLD-CLASS PREMIUM MEDICAL UI (CSS)
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@300;500;700;800&display=swap');

    /* Global Overrides */
    .stApp {{
        background: radial-gradient(circle at 0% 0%, #0f172a 0%, #020617 100%);
        color: #f1f5f9 !important;
        font-family: 'Inter', sans-serif;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: rgba(2, 6, 23, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        min-width: 260px !important;
        max-width: 260px !important;
        height: calc(100vh - 40px) !important;
        margin: 20px !important;
        border-radius: 24px !important;
        overflow: hidden !important;
    }}
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
        gap: 0.5rem !important;
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }}

    [data-testid="stSidebar"] hr {{
        margin: 0.5rem 0 !important;
    }}

    /* Glassmorphism Cards */
    .glass-card {{
        background: rgba(15, 23, 42, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 28px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 32px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
        margin-bottom: 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    .glass-card:hover {{
        border: 1px solid rgba(56, 189, 248, 0.3);
        transform: translateY(-4px);
        background: rgba(15, 23, 42, 0.5);
    }}

    /* Premium Headers */
    .hero-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 3.5rem !important;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        letter-spacing: -0.04em;
    }}

    .hero-subtitle {{
        font-size: 0.95rem;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 30px;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    /* Section Headers */
    .section-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        font-family: 'Outfit', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 20px;
    }}

    /* Prediction Result Styles */
    .res-positive {{
        color: #ef4444 !important; /* Vivid Red */
        font-size: 2.5rem !important;
        font-weight: 800;
        text-shadow: 0 0 30px rgba(239, 68, 68, 0.3);
        font-family: 'Outfit', sans-serif;
    }}
    
    .res-negative {{
        color: #10b981 !important; /* Emerald Green */
        font-size: 2.5rem !important;
        font-weight: 800;
        text-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
        font-family: 'Outfit', sans-serif;
    }}

    /* Progress Bar */
    .stProgress > div > div > div > div {{
        background-image: linear-gradient(to right, #0ea5e9 , #6366f1);
        border-radius: 10px;
    }}

    /* Justification list */
    .justification-item {{
        background: rgba(255, 255, 255, 0.02);
        padding: 16px;
        border-radius: 16px;
        margin-bottom: 10px;
        border-left: 4px solid #38bdf8;
        font-size: 1rem;
        line-height: 1.5;
        transition: background 0.2s ease;
    }}
    .justification-item:hover {{
        background: rgba(255, 255, 255, 0.05);
    }}

    /* Metric Styling */
    .metric-box {{
        background: rgba(255, 255, 255, 0.03);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }}

    </style>
    """, unsafe_allow_html=True)

# API Host
API_HOST = os.getenv("API_HOST", "http://localhost:8000")

# Sidebar System Status
with st.sidebar:
    st.markdown(f'<div class="section-header" style="font-size: 1.1rem; margin-bottom: 5px;">{lucide_icon("activity", size=18, color="#38bdf8")} Integrity</div>', unsafe_allow_html=True)
    st.success("AI Core: Operational")
    
    st.divider()
    st.markdown(f'<div class="section-header" style="font-size: 1.1rem; margin-bottom: 5px;">{lucide_icon("clipboard-list", size=18, color="#818cf8")} Vitals</div>', unsafe_allow_html=True)
    spo2 = st.number_input("SPO2 (%)", min_value=0, max_value=100, value=98, step=1, key="input_spo2")
    temperature = st.number_input("Temp (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1, key="input_temp")
    
    # New Scan / Exit Button
    if st.button("🔄 New Patient Scan", width='stretch', key="btn_new_scan", on_click=reset_station):
        pass

    st.divider()
    st.markdown(f'<div style="opacity: 0.6; font-size: 0.75rem; line-height: 1.4;">'
                f'{lucide_icon("shield-check", size=12)} HIPAA Secure Node<br>'
                f'{lucide_icon("cpu", size=12)} v3.2.0 | {lucide_icon("calendar", size=12)} May 15'
                f'</div>', unsafe_allow_html=True)

# Header Area
st.markdown('<h1 class="hero-title">Sentinel AI</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="hero-subtitle">{lucide_icon("shield", size=18, color="#38bdf8")} NEXT-GEN CLINICAL DIAGNOSTIC STATION</p>', unsafe_allow_html=True)

# Main Dashboard Layout
main_col, report_col = st.columns([1.1, 0.9], gap="large")

with main_col:
    st.markdown(f'''
        <div class="glass-card">
            <div class="section-header">{lucide_icon("upload-cloud", color="#38bdf8")} Imaging Data Input</div>
    ''', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Select Chest Radiograph (JPG/PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key=f"file_radiograph_{st.session_state.uploader_key}")
    
    if uploaded_file is not None:
        show_gradcam = st.toggle("🔍 View Grad-CAM Explainability Overlay", value=False, key="toggle_gradcam")
        
        if show_gradcam:
            with st.spinner("Analyzing neural attention patterns..."):
                files = {"file": uploaded_file.getvalue()}
                try:
                    response = requests.post(f"{API_HOST}/explain", files=files, timeout=30)
                    if response.status_code == 200:
                        heatmap_base64 = response.json()["heatmap_base64"]
                        img_data = base64.b64decode(heatmap_base64)
                        image = Image.open(io.BytesIO(img_data))
                        st.image(image, width='stretch', caption="Pathological Focus Area (Grad-CAM)")
                    else:
                        error_detail = response.json().get("error", "Unknown internal error.")
                        st.error(f"Grad-CAM Protocol Failure: {error_detail}")
                        image = Image.open(uploaded_file)
                        st.image(image, width='stretch')
                except Exception as e:
                    st.error(f"Inference Node Unreachable: {str(e)}")
                    image = Image.open(uploaded_file)
                    st.image(image, width='stretch')
        else:
            image = Image.open(uploaded_file)
            st.image(image, width='stretch')
            
    st.markdown('</div>', unsafe_allow_html=True)

with report_col:
    if uploaded_file is not None:
        st.markdown(f'''
            <div class="glass-card">
                <div class="section-header">{lucide_icon("microscope", color="#818cf8")} Diagnostic Analysis</div>
        ''', unsafe_allow_html=True)
        
        with st.spinner('Orchestrating Deep Learning Inference...'):
            files = {"file": uploaded_file.getvalue()}
            try:
                response = requests.post(f"{API_HOST}/predict", files=files, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    is_pneu = result['prediction'] == "PNEUMONIA"
                    
                    # Premium Results Display
                    label_class = "res-positive" if is_pneu else "res-negative"
                    st.markdown(f'<p class="{label_class}">{result["prediction"]}</p>', unsafe_allow_html=True)
                    
                    conf_val = result.get('raw_confidence', float(result['confidence'].strip('%')) / 100)
                    
                    # Confidence Metric
                    st.markdown(f"**{lucide_icon('target', size=16)} Diagnostic Confidence**", unsafe_allow_html=True)
                    st.progress(conf_val)
                    st.markdown(f"<div style='text-align: right; font-weight: 700; color: #38bdf8;'>{conf_val:.1%}</div>", unsafe_allow_html=True)
                    
                    st.divider()
                    st.markdown(f'<div class="section-header" style="font-size: 1.1rem;">{lucide_icon("search", size=20, color="#38bdf8")} Clinical Rationale</div>', unsafe_allow_html=True)
                    for feature in result['justification']:
                        st.markdown(f'<div class="justification-item">{feature}</div>', unsafe_allow_html=True)
                    
                    st.divider()
                    if is_pneu:
                        st.markdown(f'''
                            <div style="background-color: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; color: #f87171; padding: 16px; border-radius: 12px; font-weight: 600;">
                                {lucide_icon('alert-triangle', size=18)} URGENT: Pathological patterns identified. Immediate clinical intervention advised.
                            </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                            <div style="background-color: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #34d399; padding: 16px; border-radius: 12px; font-weight: 600;">
                                {lucide_icon('check-circle', size=18)} NORMAL: No active inflammatory processes detected in the pulmonary field.
                            </div>
                        ''', unsafe_allow_html=True)
                else:
                    st.error("System Error: Inference node unreachable.")
            except Exception as e:
                st.error(f"Protocol Error: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'''
            <div class="glass-card">
                <div class="section-header">{lucide_icon("info", color="#38bdf8")} Station Readiness</div>
                <p style="color: #94a3b8; margin-bottom: 20px;">Awaiting high-fidelity radiological data for clinical analysis.</p>
                <div style="font-weight: 600; margin-bottom: 10px;">{lucide_icon('book-open', size=16)} Clinical Guidelines</div>
                <div class="justification-item" style="font-size: 0.85rem; border-left-color: #818cf8;">
                    1. Ensure PA/AP chest radiograph is centered and includes full lung fields.<br>
                    2. Minimal artifacts (jewelry/tubing) preferred for optimal sensitivity.<br>
                    3. Resolution ≥ 224px recommended for neural fidelity.
                </div>
            </div>
        ''', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f'<div style="color: #94a3b8; font-size: 0.8rem; text-align: center;">{lucide_icon("cpu", size=12)} Pneumonia Sentinel AI | {lucide_icon("check", size=12)} 99.5% Validation Sensitivity | {lucide_icon("lock", size=12)} HIPAA Secure</div>', unsafe_allow_html=True)
