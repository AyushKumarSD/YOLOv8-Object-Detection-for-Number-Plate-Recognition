import streamlit as st
from ultralytics import YOLO
from easyocr import Reader
import cv2
import numpy as np
from detect_and_recognize import detect_number_plates, recognize_number_plates

st.set_page_config(page_title="Auto NPR", page_icon=":car:", layout="wide")

st.title('Automatic Number Plate Recognition System :car:')
st.markdown("---")

uploaded_file = st.file_uploader("Upload an Image 🚀", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with st.spinner("Processing...🛠"):
        # Read the uploaded image directly
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load the YOLO model and EasyOCR reader
        model = YOLO("runs/detect/train/weights/best.pt")
        reader = Reader(['en'], gpu=True)

        # Split the page into two columns
        col1, col2 = st.columns(2)

        # Display the original image in the first column
        with col1:
            st.subheader("Original Image")
            st.image(image_rgb)

        # Detect number plates
        number_plate_list = detect_number_plates(image_rgb, model)

        if number_plate_list:
            # Recognize text from detected number plates
            number_plate_list = recognize_number_plates(image, reader, number_plate_list)

            # Display detections and results
            for box, text in number_plate_list:
                cropped_number_plate = image_rgb[box[1]:box[3], box[0]:box[2]]

                # Draw bounding boxes and detected text on the image
                cv2.rectangle(image_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(image_rgb, text, (box[0], box[3] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display detection results in the second column
                with col2:
                    st.subheader("Number Plate Detection")
                    st.image(image_rgb)

                st.subheader("Cropped Number Plate")
                st.image(cropped_number_plate, width=300)
                st.success(f"Detected Text: **{text}**")

        else:
            st.error("No number plates detected.")
else:
    st.info("Please upload an image to get started.")
