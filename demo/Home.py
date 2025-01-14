import streamlit as st
import yolov5
import cv2
import numpy as np
from PIL import Image
model = yolov5.load("checkpoints/YOLOv5.pt") 
list_id = [4, 5, 8, 9, 10, 13, 14, 15, 16]

def predict_image(image):
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
    results = model(frame)
    st.code(results)
    
    predictions = results.xyxy[0]
    for pred in predictions:
        x1, y1, x2, y2, conf, class_id = pred
        if class_id not in list_id:
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"{model.names[int(class_id)]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    return frame
  
st.title("PPE Detection - Image Demo")

uploaded_file = st.file_uploader("Chọn một ảnh", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1.image(image, caption="Ảnh tải lên", use_container_width=True)

    st.markdown("### Kết quả dự đoán")
    annotated_image = predict_image(image)
    col2.image(annotated_image, channels="BGR", use_container_width=True)