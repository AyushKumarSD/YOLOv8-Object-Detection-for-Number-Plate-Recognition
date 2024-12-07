from ultralytics import YOLO
from easyocr import Reader
import cv2
import time
import csv
import torch

CONFIDENCE_THRESHOLD = 0.4
COLOR = (0, 255, 0)

def detect_number_plates(image, model):
    start = time.time()
    detections = model.predict(image)[0].boxes.data

    if detections.shape != torch.Size([0, 6]):
        boxes = []
        for detection in detections:
            confidence = detection[4]
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue
            boxes.append([int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])])

        print(f"{len(boxes)} Number plate(s) detected.")
        end = time.time()
        print(f"Detection Time: {(end - start):.2f}s")
        return boxes
    else:
        print("No number plates detected.")
        return []

def recognize_number_plates(image, reader, number_plate_list):
    start = time.time()
    results = []

    for box in number_plate_list:
        xmin, ymin, xmax, ymax = box
        cropped_image = image[ymin:ymax, xmin:xmax]
        detection = reader.readtext(cropped_image)

        if detection:
            text = detection[0][1]
            results.append((box, text))
        else:
            results.append((box, ""))

    end = time.time()
    print(f"Recognition Time: {(end - start):.2f}s")
    return results
