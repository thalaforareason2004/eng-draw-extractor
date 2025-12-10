import streamlit as st
from PIL import Image

st.set_page_config(page_title="Engineering Drawing Echo Test", layout="wide")

st.title("Engineering Drawing Echo Test")

uploaded_file = st.file_uploader(
    "Upload an engineering drawing image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)
    st.write("Received image.")
else:
    st.write("No image uploaded yet.")
