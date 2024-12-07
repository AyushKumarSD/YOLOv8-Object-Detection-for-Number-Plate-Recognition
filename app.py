import streamlit as st
from ultralytics import YOLO
from easyocr import Reader
import cv2
import numpy as np
import tempfile

from detect_and_recognize import detect_number_plates, recognize_number_plates

st.set_page_config(page_title="Auto NPR", page_icon=":car:", layout="wide")

st.title('Automatic Number Plate Recognition System :car:')
st.markdown("---")

uploaded_file = st.file_uploader("Upload an Image or Video 🚀", type=["png", "jpg", "jpeg", "mp4"])

if uploaded_file is not None:
    with st.spinner("Processing...🛠"):
        # If the uploaded file is a video
        if uploaded_file.type == "video/mp4":
            # Save the video file to a temporary location
            video_bytes = uploaded_file.read()
            with open("temp_video.mp4", "wb") as f:
                f.write(video_bytes)
            
            cap = cv2.VideoCapture("temp_video.mp4")
            model = YOLO("best.pt")
            reader = Reader(['en'], gpu=True)

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create a VideoWriter object to save the processed video
            output_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out = cv2.VideoWriter(output_video.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            st.subheader("Processed Video")

            # Frame counter to track frames
            frame_count = 0
            frame_interval = int(fps)  # Process 6 frames per second

            # Process each frame from the video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Only process every 'frame_interval'-th frame (10 fps)
                if frame_count % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Detect number plates
                    number_plate_list = detect_number_plates(frame_rgb, model)

                    if number_plate_list:
                        # Recognize text from detected number plates
                        number_plate_list = recognize_number_plates(frame, reader, number_plate_list)

                        # Draw bounding boxes and text on the frame
                        for box, text in number_plate_list:
                            cv2.rectangle(frame_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            cv2.putText(frame_rgb, text, (box[0], box[3] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Write the processed frame to the output video
                    out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            cap.release()
            out.release()

            # Stream the processed video in the app
            st.video(output_video.name)

        # If the uploaded file is an image
        else:
            # Read the uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Load the YOLO model and EasyOCR reader
            model = YOLO("best.pt")
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
    st.info("Please upload an image or video to get started.")
