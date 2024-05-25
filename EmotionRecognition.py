import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import tempfile

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
    with st.sidebar:
        st.page_link("pages/Chart.py", label="Chart")
    
    option = st.selectbox(
        "Select media", 
        ("Webcam", "Image", "Video"), 
        index=None,
        )
    
    if option == "Webcam":
        # Run the emotion detection app
        webrtc_streamer(key="example", video_processor_factory=EmotionDetector)
    elif option == "Image":
        # Load image and detect emotions
        image_file = st.file_uploader("Upload an image", type=["jpg", "png"])
        if image_file is not None:
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
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
            
            # Convert the image to RGB (OpenCV uses BGR by default)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert the image to a PIL image
            im_pil = Image.fromarray(img_rgb)

            st.image(im_pil, use_column_width=True)
    # elif option == "Video":
    #     # Load video and detect emotions
    #     video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
    #     if video_file is not None:
    #         # Save uploaded video to a temporary file
    #         tfile = tempfile.NamedTemporaryFile(delete=False) 
    #         tfile.write(video_file.read())
            
    #         # Open the video file
    #         video_cap = cv2.VideoCapture(tfile.name)
            
    #         # Get video properties
    #         width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #         height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #         fps = video_cap.get(cv2.CAP_PROP_FPS)
            
    #         # Define codec and create VideoWriter object
    #         fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #         out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
            
    #         while video_cap.isOpened():
    #             ret, frame = video_cap.read()
    #             if not ret:
    #                 break

    #             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #             face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    #             faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                
    #             for (x, y, w, h) in faces:
    #                 roi_gray = gray[y:y + h, x:x + w]
    #                 cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    #                 prediction = model.predict(cropped_img)
    #                 maxindex = int(np.argmax(prediction))
    #                 cv2.putText(frame, emotion_dict[maxindex], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
    #             out.write(frame)
            
    #         video_cap.release()
    #         out.release()
            
    #         # Display the processed video
    #         st.video('output.mp4')

if __name__ == "__main__":
    main()
