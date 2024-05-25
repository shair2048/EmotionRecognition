# import numpy as np
# import argparse
# import cv2
# import os
# import tensorflow as tf
# from keras.models import Model
# from keras.models import load_model
# from keras.models import Sequential
# from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout

# model = tf.keras.models.load_model('model.h5')
# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


# cap = cv2.VideoCapture(0)
# while True:
    
#     ret, frame = cap.read()
#     if not ret:
#         break
#     facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
#         roi_gray = gray[y:y + h, x:x + w]
#         cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
#         prediction = model.predict(cropped_img)
#         maxindex = int(np.argmax(prediction))
#         cv2.putText(frame, emotion_dict[maxindex], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#     cv2.imshow('Video', cv2.resize(frame,(700,480),interpolation = cv2.INTER_CUBIC))
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import tensorflow as tf

# Load pre-trained model and emotion labels
model = tf.keras.models.load_model('model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Define a video transformer class for emotion detection
class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(img, emotion_dict[maxindex], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(img, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)

        return img

def main():
    # Streamlit UI
    st.title("Real-time Emotion Detection using Streamlit as Webcam")
    st.write("This app detects emotions from your webcam feed.")

    # Run the emotion detection app
    webrtc_streamer(key="example", video_processor_factory=EmotionDetector)

if __name__ == "__main__":
    main()
