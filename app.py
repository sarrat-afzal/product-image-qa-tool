import cv2
import numpy as np
from PIL import Image
from rembg import remove
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Product Image QA & Background Processing",
    page_icon="📷",
    layout="wide",
)

st.title("📷 Product Image QA & Processing Tool")
st.write(
    "Upload product images to automatically verify quality standards and generate clean backgrounds."
)


# Helper Functions
def evaluate_blur(image_bytes):
    """Calculates Laplacian variance to detect image blur."""
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    variance = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return variance


def evaluate_brightness(image_bytes):
    """Calculates average brightness of the image."""
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    return np.mean(img_gray)


# Sidebar Configuration
st.sidebar.header("QA Threshold Settings")
blur_threshold = st.sidebar.slider(
    "Blur Threshold (Lower = More Tolerant)", 10.0, 300.0, 100.0
)
brightness_min = st.sidebar.slider("Minimum Brightness", 0, 255, 40)

# File Uploader
uploaded_file = st.file_uploader(
    "Choose a product image...", type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    raw_image = Image.open(uploaded_file)

    st.subheader("1. Quality Assessment")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(raw_image, caption="Uploaded Image", use_column_width=True)

    with col2:
        # Run Quality Checks
        blur_score = evaluate_blur(bytes_data)
        brightness_score = evaluate_brightness(bytes_data)
        width, height = raw_image.size

        # Quality Metrics Display
        st.write(f"**Dimensions:** {width} x {height} px")
        st.write(f"**Blur Score:** {blur_score:.2f} *(Threshold: {blur_threshold})*")
        st.write(
            f"**Brightness Score:** {brightness_score:.2f} *(Min Required: {brightness_min})*"
        )

        # Pass / Fail Warnings
        is_passed = True
        if blur_score < blur_threshold:
            st.error("⚠️ Quality Alert: Image appears to be blurry!")
            is_passed = False
        else:
            st.success("✅ Sharpness Check Passed")

        if brightness_score < brightness_min:
            st.warning("⚠️ Quality Alert: Image is too dark!")
            is_passed = False
        else:
            st.success("✅ Brightness Check Passed")

    st.markdown("---")
    st.subheader("2. Background Processing")

    if st.button("Process & Remove Background"):
        with st.spinner("Processing background removal..."):
            # Process Background
            processed_bytes = remove(bytes_data)
            processed_image = Image.open(
                io.BytesIO(processed_bytes)
            ).convert("RGBA")

            col_a, col_b = st.columns(2)
            with col_a:
                st.image(
                    processed_image,
                    caption="Transparent Cutout",
                    use_column_width=True,
                )

            with col_b:
                # White Background Option
                white_bg = Image.new("RGBA", processed_image.size, "WHITE")
                white_bg.paste(
                    processed_image, (0, 0), processed_image
                )  # Alpha mask
                st.image(
                    white_bg,
                    caption="White E-Commerce Background",
                    use_column_width=True,
                )

            # Download Option
            st.download_button(
                label="📥 Download Clean Image (PNG)",
                data=processed_bytes,
                file_name="processed_product.png",
                mime="image/png",
            )
