import streamlit as st

st.set_page_config(page_title="Crop Disease Detection", layout="wide")

st.title("🌱 Crop Disease Detection & Annotation System")

uploaded_file = st.file_uploader(
    "Upload a Plant Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    st.subheader("📌 Annotation")
    st.info("Annotation feature will be integrated with AI prediction.")

    st.subheader("🦠 Disease Prediction")
    st.success("Model prediction will appear here.")

    st.subheader("⚠ Possible Cause")
    st.write("Possible causes of the detected disease will be shown here.")

    st.subheader("💡 Recommendation")
    st.write("Recommended treatment and management practices.")

    st.subheader("✅ Suggestions")
    st.write("Preventive measures and future care suggestions.")

st.sidebar.title("History")
st.sidebar.info("Uploaded cases history will be displayed here.")
