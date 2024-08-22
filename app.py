import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

def predict(chosen_model, img, classes=[], conf=0.5):
    """
    Runs object detection on the image using the chosen model.

    Args:
        chosen_model: YOLO model instance
        img: Input image (numpy array)
        classes: List of class IDs to filter predictions
        conf: Confidence threshold for predictions

    Returns:
        Results object from YOLO model
    """
    if classes:
        results = chosen_model(img, classes=classes, conf=conf)
    else:
        results = chosen_model(img, conf=conf)
    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    """
    Runs object detection and draws bounding boxes on the image.

    Args:
        chosen_model: YOLO model instance
        img: Input image (numpy array)
        classes: List of class IDs to filter predictions
        conf: Confidence threshold for predictions

    Returns:
        Annotated image and detection results
    """
    img_copy = img.copy()  # Create a copy of the original image
    results = predict(chosen_model, img_copy, classes, conf)
    
    # Iterate over results
    for result in results:
        for box in result.boxes:
            # Convert box coordinates and draw rectangles
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = result.names[cls] if cls < len(result.names) else "Unknown"

            # Draw bounding box and label
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_copy, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    
    return img_copy, results

# Load the trained YOLOv8 model
model = YOLO('best.pt')  # Replace with the path to your best model

# Title of the Streamlit app
st.title('Where\'s Waldo - Object Detection')

# File uploader for users to upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)  # Display the image
    st.write("Detecting Waldo...")

    # Convert the image to a format compatible with OpenCV
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Run detection and draw bounding boxes
    annotated_image, _ = predict_and_detect(model, image_bgr, conf=0.5)

    # Convert the image back to RGB format for PIL
    image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    image_final = Image.fromarray(image_rgb)

    # Display the image with bounding boxes in Streamlit
    st.image(image_final, caption='Detected Waldo', use_column_width=True)
