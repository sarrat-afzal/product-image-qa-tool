import io
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Product Image QA & Background Suite",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📸 E-Commerce Image QA & Background Processing Pipeline")
st.markdown(
    "Automated computer vision pipeline to inspect image clarity, detect poor lighting, and isolate product subjects for e-commerce catalogs."
)

# Sidebar Configuration
st.sidebar.header("⚙️ Quality Control Settings")
blur_threshold = st.sidebar.slider(
    "Sharpness Threshold (Laplacian Var)",
    min_value=10.0,
    max_value=300.0,
    value=100.0,
    help="Higher values demand sharper focus.",
)
min_brightness = st.sidebar.slider(
    "Minimum Brightness (0-255)",
    min_value=10,
    max_value=150,
    value=45,
    help="Images darker than this threshold will be flagged.",
)
bg_choice = st.sidebar.selectbox(
    "Output Background Style",
    ["Transparent (PNG)", "Clean White Studio (JPEG)"],
)


# Core Processing Functions
def analyze_image_quality(image_bytes):
    """Computes blur variance and mean brightness using OpenCV."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    gray = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    # Blur detection via Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Brightness calculation
    brightness = np.mean(gray)

    return laplacian_var, brightness


def remove_background(image_bytes, output_style):
    """Applies U-2-Net model via rembg to extract subject."""
    cutout_bytes = remove(image_bytes)
    cutout_img = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")

    if output_style == "Clean White Studio (JPEG)":
        white_canvas = Image.new("RGB", cutout_img.size, (255, 255, 255))
        white_canvas.paste(cutout_img, mask=cutout_img.split()[3])
        buf = io.BytesIO()
        white_canvas.save(buf, format="JPEG", quality=95)
        return buf.getvalue(), "image/jpeg", "jpg"
    else:
        return cutout_bytes, "image/png", "png"


# Mode Selection
tab_single, tab_batch = st.tabs(
    ["Single Image Inspection", "Batch E-Commerce Processing"]
)

with tab_single:
    uploaded_file = st.file_uploader(
        "Upload a product image",
        type=["png", "jpg", "jpeg", "webp"],
        key="single",
    )

    if uploaded_file:
        raw_bytes = uploaded_file.getvalue()
        raw_image = Image.open(uploaded_file)
        w, h = raw_image.size

        # Analyze Quality
        blur_score, brightness_score = analyze_image_quality(raw_bytes)
        is_sharp = blur_score >= blur_threshold
        is_bright = brightness_score >= min_brightness
        is_qa_pass = is_sharp and is_bright

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(
                raw_image,
                caption=f"Original ({w}x{h} px)",
                use_container_width=True,
            )

        with col2:
            st.subheader("QA Inspection Diagnostics")

            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Sharpness Score", f"{blur_score:.1f}")
            metric_col2.metric("Brightness Level", f"{brightness_score:.1f}")

            if is_qa_pass:
                st.success("✅ Image passes all quality standards.")
            else:
                if not is_sharp:
                    st.error("⚠️ Blurry Image Detected: Subject lacks crispness.")
                if not is_bright:
                    st.warning("⚠️ Low Lighting: Image falls below brightness requirement.")

            st.markdown("---")
            if st.button("Generate Clean Asset", type="primary"):
                with st.spinner("Executing neural background removal..."):
                    out_bytes, mime, ext = remove_background(
                        raw_bytes, bg_choice
                    )
                    st.image(
                        out_bytes,
                        caption=f"Result ({bg_choice})",
                        use_container_width=True,
                    )
                    st.download_button(
                        label=f"📥 Download Processed Image (.{ext})",
                        data=out_bytes,
                        file_name=f"processed_product.{ext}",
                        mime=mime,
                    )

with tab_batch:
    uploaded_files = st.file_uploader(
        "Upload multiple product images for catalog processing",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="batch",
    )

    if uploaded_files:
        st.write(f"Loaded {len(uploaded_files)} images.")
        if st.button("Run Batch Pipeline", type="primary"):
            progress_bar = st.progress(0)
            results_cols = st.columns(3)

            for idx, file in enumerate(uploaded_files):
                f_bytes = file.getvalue()
                blur, _ = analyze_image_quality(f_bytes)
                processed, mime, ext = remove_background(f_bytes, bg_choice)

                col = results_cols[idx % 3]
                with col:
                    col.image(
                        processed,
                        caption=f"{file.name} (Sharpness: {blur:.0f})",
                        use_container_width=True,
                    )
                    col.download_button(
                        label=f"Download {file.name}",
                        data=processed,
                        file_name=f"clean_{file.name.rsplit('.', 1)[0]}.{ext}",
                        mime=mime,
                        key=f"dl_{idx}",
                    )

                progress_bar.progress((idx + 1) / len(uploaded_files))
            st.success("Batch processing complete!")
