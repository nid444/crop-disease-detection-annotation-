import streamlit as st
import gdown
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crop Annotation",
    page_icon="🌿",
    layout="wide"
)

# ---------------- PREMIUM CSS ----------------
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: white;
    }

    h1 {
        color: #4ade80;
        text-align: center;
        font-size: 44px;
        font-weight: 800;
    }

    h2, h3 {
        color: #a7f3d0;
    }

    .stButton>button {
        background-color: #22c55e;
        color: white;
        border-radius: 12px;
        height: 50px;
        width: 220px;
        font-size: 18px;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #16a34a;
    }

    .block {
        background: #1e293b;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1>🌿 Crop Annotation</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:18px;'>AI-powered Plant Leaf Disease Detection System</p>", unsafe_allow_html=True)

# ---------------- MODEL DOWNLOAD ----------------
file_id = "19xEmlzNGU8UDXfBKXkICj3_bhwvIDtFm"
model_path = "plant_disease_model.h5"

if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

model = load_model(model_path)

# ---------------- CLASS LABELS ----------------
classes = [
    "Healthy",
    "Leaf Blight",
    "Rust Disease",
    "Bacterial Spot"
]

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Dashboard")
st.sidebar.info("Crop Annotation AI System")
st.sidebar.success("Model loaded from Google Drive")

history = []

# ---------------- MAIN UI ----------------
col1, col2 = st.columns([1, 1])

# ---------------- LEFT SIDE ----------------
with col1:
    st.subheader("📤 Upload Plant Leaf Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

# ---------------- RIGHT SIDE ----------------
with col2:
    st.subheader("🧠 AI Prediction Panel")

    if uploaded_file is not None:

        st.markdown("Click below to analyze the plant leaf")

        if st.button("Predict Disease 🚀"):

            # preprocessing
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # prediction
            prediction = model.predict(img_array)
            result = classes[np.argmax(prediction)]

            st.markdown("### 🦠 Disease Prediction")
            st.success(f"Prediction: {result}")

            # history store
            history.append(result)

            # ---------------- CAUSE ----------------
            st.markdown("### ⚠ Possible Cause")

            if result == "Healthy":
                st.info("Plant is healthy. No disease detected.")
            elif result == "Leaf Blight":
                st.info("Caused by fungal infection and high humidity.")
            elif result == "Rust Disease":
                st.info("Caused by fungal spores in moist environment.")
            else:
                st.info("Caused by bacterial infection and poor plant hygiene.")

            # ---------------- RECOMMENDATION ----------------
            st.markdown("### 💡 Recommendation")

            if result == "Healthy":
                st.info("Continue regular care and monitoring.")
            else:
                st.info("Apply fungicide/bactericide and remove infected leaves.")

            # ---------------- SUGGESTIONS ----------------
            st.markdown("### ✅ Suggestions")
            st.info("Maintain proper watering, spacing, and plant hygiene.")

# ---------------- HISTORY SECTION ----------------
st.sidebar.markdown("### 🧾 History (Session)")
if uploaded_file is not None:
    st.sidebar.write(history)
