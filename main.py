import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import base64

# -----------------------------------
# Background Image Function
# -----------------------------------
def add_bg_from_local(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded}");
                background-size: cover;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Background image not found!")


# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Plane Classification",
    layout="centered",
    page_icon="🧠"
)

# Add background
add_bg_from_local(r"E:\Projects\Computer VIsion\Brain_Tumor_Detection\background.jpg")

st.title("🧠 Brain MRI Plane Classification")
st.write("Upload an MRI image to detect whether it is **Coronal**, **Axial**, or **Sagittal**.")


# -----------------------------------
# Load YOLO Model
# -----------------------------------
MODEL_PATH = r"E:\Projects\Computer VIsion\Brain_Tumor_Detection\brain_tumor_detection.pt"

if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    st.error("Model not found! Please check your model path.")


# ===================================
# --- Prediction Function -----------
# ===================================
def predict_and_display(model, uploaded_file):
    # Load and convert image
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    st.subheader("📤 Uploaded Image")
    st.image(img, use_column_width=True)

    # Run YOLO prediction
    results = model(img_array)[0]

    # Class names
    CLASS_MAP = {0: "Coronal", 1: "Axial", 2: "Sagittal"}

    # If NO detections
    if len(results.boxes) == 0:
        st.error("No MRI plane detected!")
        return

    # ----------------------------------------
    #  GET ONLY the highest-confidence result
    # ----------------------------------------
    max_conf_index = np.argmax(results.boxes.conf.cpu().numpy())
    best_box = results.boxes.xyxy[max_conf_index]
    best_conf = results.boxes.conf[max_conf_index]
    best_cls = int(results.boxes.cls[max_conf_index])

    x1, y1, x2, y2 = map(int, best_box)
    label = CLASS_MAP.get(best_cls, f"Class {best_cls}")

    # Prepare two images
    output_img = img_array.copy()
    overlay = img_array.copy()

    # Random color
    color = tuple(np.random.randint(0, 255, 3).tolist())

    # Draw box
    cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 3)

    # Draw label
    cv2.putText(
        output_img,
        f"{label} {best_conf:.2f}",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    # Transparent overlay mask
    overlay[y1:y2, x1:x2] = (
        overlay[y1:y2, x1:x2] * 0.4 + np.array(color) * 0.6
    ).astype(np.uint8)

    # Final blended image
    final_img = cv2.addWeighted(overlay, 0.4, output_img, 0.6, 0)

    st.subheader(" Predicted Result")
    st.image(final_img, caption=f"Detected Plane: {label}", use_column_width=True)


# -----------------------------------
# File Uploader
# -----------------------------------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    predict_and_display(model, uploaded_file)
