import streamlit as st
from ultralytics import YOLO
from easyocr import Reader
import cv2
import numpy as np

st.set_page_config(page_title="Auto NPR", page_icon=":car:", layout="wide")

st.title('Automatic Number Plate Recognition System :car:')
st.markdown("---")

uploaded_file = st.file_uploader("Upload an Image 🚀", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with st.spinner("In progress ...🛠"):
        # Read the uploaded image directly without saving
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load the YOLO model
        model = YOLO("../runs/detect/train/weights/best.pt")
        # Initialize the EasyOCR reader
        reader = Reader(['en'], gpu=True)

        # Make a copy of the image to draw on it
        image_copy = image_rgb.copy()
        # Split the page into two columns
        col1, col2 = st.columns(2)
        
        # Display the original image in the first column
        with col1:
            st.subheader("Original Image")
            st.image(image_rgb)

        from detect_and_recognize import detect_number_plates, recognize_number_plates
        
        # Detect number plates
        number_plate_list = detect_number_plates(image_rgb, model)

        if number_plate_list:
            # Recognize number plates
            number_plate_list = recognize_number_plates(uploaded_file, reader, number_plate_list)

            for box, text in number_plate_list:
                cropped_number_plate = image_copy[box[1]:box[3], box[0]:box[2]]

                cv2.rectangle(image_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(image_rgb, text, (box[0], box[3] + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display the number plate detection in the second column
                with col2:
                    st.subheader("Number Plate Detection")
                    st.image(image_rgb)

                st.subheader("Cropped Number Plate")
                st.image(cropped_number_plate, width=300)
                st.success("Number plate text: **{}**".format(text))

        else:
            st.error("No number plate detected.")

else:
    st.info("Please upload an image to get started.")
