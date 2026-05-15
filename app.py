import streamlit as st
import httpx
from PIL import Image, ImageFilter
import io
import base64
import numpy as np
import cv2
import time
import json
import logging
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SENTINEL V1.0 PRODUCTION RELEASE ---
# ===================== CONFIG =====================
st.set_page_config(page_title="Sentinel-Ai Clinical Suite v1.0", page_icon="🛡️", layout="wide")
API_URL = "http://localhost:8000"

# ===================== CLINICAL CSS (v1.0 INFINITY) =====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Outfit', sans-serif; }
    /* Comfort-Fit Sidebar Logic */
    [data-testid="stSidebar"] {
        width: 280px !important;
        min-width: 280px !important;
    }
    [data-testid="stSidebar"] .st-emotion-cache-1647n7p { padding: 1rem 1rem; }

    .diagnostic-badge {
        padding: 5px 15px; border-radius: 8px; text-align: center; margin-bottom: 8px;
        backdrop-filter: blur(15px); border: 1px solid rgba(255,255,255,0.1);
    }
    .badge-positive { background: rgba(230, 57, 70, 0.2); border-color: #E63946; }
    .badge-normal { background: rgba(42, 157, 143, 0.2); border-color: #2A9D8F; }

    .glass-card {
        background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px; padding: 15px; margin-bottom: 15px;
    }
    .hud-label { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: #64748b; letter-spacing: 2px; }
    .hud-value { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #00d4ff; }
</style>
""", unsafe_allow_html=True)

# ===================== API HELPERS =====================
async def call_api_predict(image_bytes, enhance=False):
    async with httpx.AsyncClient() as client:
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        params = {'enhance': enhance}
        try:
            response = await client.post(f"{API_URL}/predict", files=files, params=params, timeout=30.0)
            return response.json()
        except: return {"status": "offline"}

async def call_api_explain(image_bytes):
    async with httpx.AsyncClient() as client:
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        try:
            response = await client.post(f"{API_URL}/explain", files=files, timeout=45.0)
            return response.json()
        except: return {"status": "offline"}

# ===================== SIDEBAR (COMMAND HUB) =====================
if "reset_counter" not in st.session_state: st.session_state.reset_counter = 0

with st.sidebar:
    st.markdown("<h1 style='color: #00d4ff; font-weight: 800; margin:0; font-size: 1.2rem;'>SENTINEL-Ai V1</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.6rem; color: #64748b; margin-bottom: 10px;'>PRODUCTION RELEASE</p>", unsafe_allow_html=True)

    st.markdown("<p class='hud-label'>🏥 CONFIGURATION</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: facility = st.text_input("Facility", "CENTRAL", key="facility", label_visibility="collapsed")
    with c2: clinician_id = st.text_input("ID", "STN-77", key="clinician", label_visibility="collapsed")

    st.markdown("<p class='hud-label'>⚙️ ENGINE CONTROLS</p>", unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1: enhance_mode = st.toggle("Enhance", value=False)
    with e2: privacy_shield = st.toggle("Shield", value=True)
    show_gcam = st.toggle("Neural Focus (Grad-CAM)", value=True)

    cam_opacity = st.slider("Heatmap Opacity", 0.0, 1.0, 0.6)

    if st.button("🔄 REBOOT", use_container_width=True):
        st.session_state.reset_counter += 1
        for key in list(st.session_state.keys()):
            if key != "reset_counter": del st.session_state[key]
        st.rerun()

    st.markdown("<div style='border-top: 1px solid rgba(255,255,255,0.05); margin-top: 20px; padding-top: 15px;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size: 0.7rem; color: #64748b; line-height: 1.4;'>
        <b style='color: #94a3b8;'>SYSTEM SPECIFICATIONS</b><br>
        Core: ResNet-50 x OpenVINO<br>
        Metrics: 99.5% Sen | 94.07% Acc<br><br>
        <b style='color: #94a3b8;'>CONTACT SUPPORT</b><br>
        <a href='mailto:snyper191@gmail.com' style='color: #00d4ff; text-decoration: none;'>snyper191@gmail.com</a>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ===================== MAIN STAGE =====================
st.markdown("<div class='glass-card' style='margin-bottom: 20px;'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📥 DEPLOY RADIOGRAPH FOR ANALYSIS", type=["jpg","jpeg","png"], key=f"main_{st.session_state.reset_counter}")
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    # Diagnostic Header
    st.markdown(f"""
        <div class='glass-card' style='display: flex; justify-content: space-between; align-items: center;'>
            <div><span class='hud-label'>REF:</span> <span class='hud-value'>{uploaded_file.name}</span></div>
            <div class='hud-label' style='color: #2A9D8F;'>CONSENSUS: NEURAL VOTING ACTIVE</div>
        </div>
    """, unsafe_allow_html=True)

    img_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    # Apply PII Shield
    if privacy_shield:
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        cv2.rectangle(img_np, (0, 0), (int(w*0.3), int(h*0.1)), (0, 0, 0), -1) # Top Left
        cv2.rectangle(img_np, (int(w*0.7), 0), (w, int(h*0.1)), (0, 0, 0), -1) # Top Right
        image = Image.fromarray(img_np)
        img_bytes_shielded = io.BytesIO()
        image.save(img_bytes_shielded, format='JPEG')
        proc_bytes = img_bytes_shielded.getvalue()
    else:
        proc_bytes = img_bytes

    import asyncio
    with st.spinner("INITIATING INFINITY TRIAGE..."):
        async def analyze():
            tasks = [call_api_predict(proc_bytes, enhance=enhance_mode)]
            if show_gcam: tasks.append(call_api_explain(proc_bytes))
            return await asyncio.gather(*tasks)

        results = asyncio.run(analyze())
        predict_res = results[0]
        explain_res = results[1] if show_gcam else {"status": "disabled"}

    if predict_res.get("status") == "success":
        prediction = predict_res["prediction"]
        confidence = predict_res["raw_confidence"]
        justifications = predict_res["justification"]

        # 1. Visualization
        if show_gcam:
            col_raw, col_cam = st.columns(2)
            with col_raw:
                st.markdown("<p class='hud-label' style='text-align: center;'>ENHANCED RADIOGRAPH</p>", unsafe_allow_html=True)
                st.image(image, use_container_width=True)
            with col_cam:
                st.markdown("<p class='hud-label' style='text-align: center;'>NEURAL FOCUS (GRAD-CAM)</p>", unsafe_allow_html=True)
                if explain_res.get("status") == "success":
                    hm_bytes = base64.b64decode(explain_res["heatmap_base64"])
                    hm_np = cv2.imdecode(np.frombuffer(hm_bytes, np.uint8), cv2.IMREAD_COLOR)
                    hm_np = cv2.cvtColor(hm_np, cv2.COLOR_BGR2RGB)
                    orig_np = np.array(image.resize((224, 224)))
                    hm_np = cv2.resize(hm_np, (224, 224))
                    overlay = cv2.addWeighted(orig_np, 1 - cam_opacity, hm_np, cam_opacity, 0)
                    st.image(overlay, use_container_width=True)
                else: st.error("CAM Logic Unavailable")
        else:
            st.image(image, use_container_width=True)

        # 2. Result Ribbon
        outcome_class = "badge-positive" if prediction == "PNEUMONIA" else "badge-normal"
        headline_class = "headline-red" if prediction == "PNEUMONIA" else "headline-green"
        st.markdown(f"""
            <div class='diagnostic-badge {outcome_class}'>
                <span class='{headline_class}' style='font-size: 1.2rem; font-weight: 800;'>{prediction}</span>
                <span style='color: #94a3b8; font-size: 0.75rem; margin-left: 20px; font-family: monospace;'>CONFIDENCE CONSENSUS: {confidence:.2%}</span>
            </div>
        """, unsafe_allow_html=True)

        # 4. Findings & Report
        col_just, col_feed = st.columns([2, 1])
        with col_just:
            st.markdown("<div class='glass-card' style='padding: 12px;'>", unsafe_allow_html=True)
            st.markdown("<span class='hud-label'>AI CLINICAL RATIONALE</span>", unsafe_allow_html=True)
            for f in justifications:
                st.markdown(f"<p style='color: #cbd5e1; font-size: 0.8rem; margin: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.01);'>- {f}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_feed:
            st.markdown("<div class='glass-card' style='text-align: center;'>", unsafe_allow_html=True)
            st.markdown("<span class='hud-label'>CLINICAL DOCUMENTATION</span>", unsafe_allow_html=True)
            try:
                if FPDF is None:
                    st.error("PDF Library Missing - Please Restart App")
                else:
                    # Initialize with explicit A4 Geometry
                    pdf = FPDF(orientation='P', unit='mm', format='A4')
                    pdf.set_margins(15, 15, 15)
                    pdf.add_page(); pdf.set_font("Helvetica", "B", 16)
                    pdf.cell(0, 10, "SENTINEL-Ai V1.0 OFFICIAL CLINICAL SUMMARY", ln=True, align="C"); pdf.ln(10)

                    f_name = str(st.session_state.get("facility", "SENTINEL CENTRAL")).upper()
                    c_id = str(st.session_state.get("clinician", "STN-77"))

                    pdf.set_font("Helvetica", "", 10)
                    pdf.cell(0, 10, f"Facility: {f_name}", ln=True)
                    pdf.cell(0, 10, f"Clinician: {c_id}", ln=True)
                    pdf.cell(0, 10, f"Outcome: {str(prediction)} ({confidence:.2%})", ln=True)
                    pdf.ln(10); pdf.set_font("Helvetica", "B", 12); pdf.cell(0, 10, "FINDINGS:", ln=True)

                    pdf.set_font("Helvetica", "", 10)
                    for f in justifications:
                        pdf.multi_cell(w=180, h=8, txt=f"- {str(f)}")
                        pdf.ln(1)

                    # High-Stability Buffer Protocol
                    pdf_bytes = pdf.output()
                    if isinstance(pdf_bytes, bytearray):
                        pdf_bytes = bytes(pdf_bytes)
                    elif isinstance(pdf_bytes, str):
                        pdf_bytes = pdf_bytes.encode('latin-1')

                    st.download_button(
                        label="📄 DOWNLOAD V1 REPORT",
                        data=pdf_bytes,
                        file_name=f"Sentinel_Report_V1_{c_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Report Engine Error: {str(e)}")
                logger.error(f"PDF Error: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
        <div style='position: fixed; bottom: 0; width: 100%; background: #030508; padding: 10px; border-top: 1px solid rgba(255,255,255,0.05);'>
            <span class='hud-label'>CORE:</span> <span class='hud-value'>SENTINEL-Ai-v1.0.0</span>
            <span style='margin-left: 20px;' class='hud-label'>BIAS:</span> <span class='hud-value'>MAX-SENSITIVITY</span>
            <span style='margin-left: 20px;' class='hud-label'>ENHANCE:</span> <span class='hud-value'>{'CLAHE ACTIVE' if enhance_mode else 'RAW'}</span>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""<div style='text-align: center; margin-top: 150px;'><h1 style='color: rgba(255,255,255,0.03); font-size: 8rem; font-weight: 800; margin:0;'>SENTINEL-Ai</h1><p style='color: #2d3748; letter-spacing: 15px;'>V1.0.0 PRODUCTION RELEASE</p></div>""", unsafe_allow_html=True)