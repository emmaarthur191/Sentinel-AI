# Sentinel AI | Clinical Pneumonia Detection Suite

**Sentinel AI** is an advanced, high-performance radiological diagnostic station designed to bridge the gap between deep learning research and clinical utility. Optimized for **Intel hardware**, it provides real-time detection of pneumonia with exceptional sensitivity and explainable neural rationale.

---

### Performance Metrics (Clinical Gold Standard)
Validated against the pediatric chest radiograph dataset with rigorous evaluation metrics:
*   **Pneumonia Sensitivity (Recall):** `99.5%` (Caught 385/390 active cases)
*   **Overall Accuracy:** `94.07%`
*   **Precision (Normal):** `98%`
*   **AUC-ROC:** `0.987`
*   **MCC (Matthews Correlation Coefficient):** `0.874`

---

### Explainable AI (XAI) for Clinicians
The system provides more than just a binary classification; it offers **Diagnostic Justification** to support clinical decision-making:
*   **Neural Attention Mapping:** Real-time **Grad-CAM** overlays highlighting the specific pathological region of interest.
*   **Clinical Rationale Engine:** A dynamic narrative system that translates neural confidence into descriptive radiological findings (e.g., pulmonary opacification, consolidation patterns).

---

### Technology Stack
*   **Core Engine:** PyTorch (ResNet50 Architecture)
*   **Optimization:** Intel OpenVINO™ (Intermediate Representation)
*   **Inference Hardware:** Intel Iris Xe GPU / CPU
*   **Backend:** FastAPI (Asynchronous Inference Node)
*   **Frontend:** Streamlit (Next-Gen Clinical Dashboard)
*   **Containerization:** Docker & Docker-Compose

---

### Repository Structure
*   `api.py`: High-performance REST API with security headers and OpenVINO integration.
*   `app.py`: The "Doctor's Dashboard" featuring glassmorphism UI and real-time reporting.
*   `pneumonia_predictor.py`: Core training and evaluation logic.
*   `best_pneumonia_model_openvino.xml/bin`: Intel-optimized production weights.
*   `docker-compose.yml`: Full-stack orchestration for clinical deployment.

---

### Installation & Deployment

#### 1. Containerized (Recommended)
Ensure Docker Desktop is running, then execute:
```bash
docker-compose up --build
```
*   **Diagnostic Portal:** `http://localhost:8501`
*   **API Interface:** `http://localhost:8000/docs`

#### 2. Manual Setup
```bash
conda activate intel_open_veno
pip install -r requirements-api.txt
pip install -r requirements-app.txt
```
Run in separate terminals:
1. `python api.py`
2. `python -m streamlit run app.py`

---

### Dataset Attribution
This project utilizes the **[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** dataset provided by Paul Mooney. It consists of pediatric Anterior-Posterior radiograph images.

---

**Disclaimer:** This tool is an AI-assisted diagnostic aid and must be used in conjunction with professional clinical judgment.
