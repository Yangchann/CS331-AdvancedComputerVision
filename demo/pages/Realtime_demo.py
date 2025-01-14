import streamlit as st
import cv2
import tempfile
import numpy as np
import yolov5

list_id = [4, 5, 8, 9, 10, 13, 14, 15, 16]
model = yolov5.load("checkpoints/YOLOv5.pt") 
want_break = False
def break_loop():
    global want_break
    want_break = True

def predict_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Không thể mở camera!")
        return
    
    stresult = st.empty()
    stframe = st.empty()  

    while True:
      ret, frame = cap.read()
      if not ret:
          st.warning("Không nhận được frame từ camera!")
          break
        
      results = model(frame)  
      predictions = results.xyxy[0]
      for pred in predictions:
          x1, y1, x2, y2, conf, class_id = pred
          if class_id not in list_id:
            continue
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
          cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  
          cv2.putText(frame, f"{model.names[int(class_id)]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
          
      stframe.image(
          frame, channels="BGR", use_container_width=True
      )
      stresult.code(results)
      
      if want_break:
        break
    cap.release()

st.title("PPE Detection - Realtime Demo")
if st.button("Bắt đầu", key="start"):
    st.button("Dừng", key="stop", on_click=break_loop)
    predict_video()
