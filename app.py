import streamlit as st
from PIL import Image

from yolo_model import run_yolo_on_page


st.set_page_config(page_title="Engineering Drawing YOLO Detector", layout="wide")

st.title("Engineering Drawing YOLO Detector")

uploaded_file = st.file_uploader(
    "Upload an engineering drawing image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original image")
    st.image(image, use_column_width=True)

    with st.spinner("Running YOLO detection..."):
        try:
            result = run_yolo_on_page(image, conf_threshold=0.3)
        except Exception as e:
            st.error(f"Error running YOLO: {e}")
            st.stop()

    annotated = result.get("annotated_image")
    crops = result.get("crops", [])

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Annotated detections")
        if annotated is not None:
            st.image(annotated, use_column_width=True)
        else:
            st.write("No annotated image returned.")

    with col2:
        st.subheader("Detections summary")
        if not crops:
            st.write("No detections.")
        else:
            st.write(f"Total detections: {len(crops)}")
            st.write("---")
            for i, c in enumerate(crops):
                st.write(
                    f"[{i}] class={c['cls_name']}  "
                    f"conf={c['conf']:.2f}  box={c['box']}"
                )

else:
    st.write("No image uploaded yet.")
