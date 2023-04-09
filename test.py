import streamlit as st
import cv2
import numpy as np
from pyvirtualcam import Camera

st.subheader('Phát hiện khuôn mặt')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

with Camera(width=640, height=480, fps=20) as cam:
    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam.send(frame)
            st.image(frame)
        else:
            break

cap.release()
