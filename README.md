# Product Image QA & Background Removal Suite

An automated Computer Vision pipeline designed for e-commerce ingestion workflows. It evaluates incoming catalog imagery for sharpness and lighting compliance, followed by deep-learning background isolation.

## Architecture & Features
- **Sharpness Diagnostics:** Evaluates image high-frequency edges using Laplacian variance calculation.
- **Lighting Analysis:** Inspects pixel luminance distributions to flag underexposed product photos.
- **Deep-Learning Background Extraction:** Built on U-2-Net (`rembg`) for semantic foreground segmentation.
- **Batch Processing Support:** Multi-file ingestion and export for catalog workflows.
- **Deployable & Containerized:** Includes a lightweight `Dockerfile` for production server deployment.

## Tech Stack
- **Framework:** Streamlit
- **CV & Processing:** OpenCV, Pillow, NumPy
- **Neural Model:** rembg (U-2-Net)

## Local Run Instructions
```bash
pip install -r requirements.txt
streamlit run app.py
