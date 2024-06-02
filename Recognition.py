import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import threading
import time
import matplotlib.pyplot as plt
import tempfile

# Load pre-trained model and emotion labels
model = tf.keras.models.load_model('model_fer.h5')
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# Global array to store emotion times from each session
if "all_emotion_times" not in st.session_state:
    st.session_state.all_emotion_times = []

# Define a video transformer class for emotion detection
class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.emotion_times = {emotion: 0 for emotion in emotion_dict.values()}
        self.last_emotion = None
        self.last_update_time = time.time()
        self.lock = threading.Lock()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        current_time = time.time()
        with self.lock:
            if self.last_emotion is not None:
                self.emotion_times[self.last_emotion] += current_time - self.last_update_time

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                    prediction = model.predict(cropped_img)
                    maxindex = int(np.argmax(prediction))
                    emotion = emotion_dict[maxindex]

                    if emotion != self.last_emotion:
                        self.last_emotion = emotion
                        self.last_update_time = current_time

                    prob = np.max(prediction)
                    cv2.putText(img, f"{emotion} ({prob:.2f})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(img, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                    break  # We only care about the first detected face

            self.last_update_time = current_time

        return img
    
    def transformVideo(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        current_time = time.time()
        with self.lock:
            if self.last_emotion is not None:
                self.emotion_times[self.last_emotion] += current_time - self.last_update_time

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            emotion = emotion_dict[maxindex]

            if emotion != self.last_emotion:
                self.last_emotion = emotion
                self.last_update_time = current_time

            prob = np.max(prediction)
            cv2.putText(frame, f"{emotion} ({prob:.2f})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            break  # We only care about the first detected face

        self.last_update_time = current_time

        return frame


    def reset_emotion_times(self):
        with self.lock:
            self.emotion_times = {emotion: 0 for emotion in emotion_dict.values()}
            self.last_emotion = None
            self.last_update_time = time.time()

    def get_emotion_times(self):
        with self.lock:
            current_time = time.time()
            if self.last_emotion is not None:
                self.emotion_times[self.last_emotion] += current_time - self.last_update_time
            self.last_update_time = current_time
            
        # Định dạng các giá trị của emotion_times để có 2 chữ số sau dấu phẩy
        formatted_emotion_times = {emotion: f"{time_spent:.2f}" for emotion, time_spent in self.emotion_times.items()}
        
        return formatted_emotion_times

def plot_emotion_pie_chart(emotion_history):
    total_times = {emotion: 0 for emotion in emotion_dict.values()}
    for session in emotion_history:
        for emotion, time in session.items():
            total_times[emotion] += float(time)  # Chuyển đổi thành số thực trước khi tính tổng

    # Filter out emotions with zero time
    total_times = {emotion: time for emotion, time in total_times.items() if time > 0}

    # Plot the pie chart
    fig, ax = plt.subplots(facecolor='none')  # Đặt màu nền thành màu trong suốt
    ax.pie(total_times.values(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Thu nhỏ biểu đồ
    fig.set_size_inches(3, 3)  # Đặt kích thước của hình thành 4x4 inches

    # Thêm bảng chú thích màu
    plt.legend(total_times.keys(), loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small')

    st.pyplot(fig)


def main():
    st.set_page_config(page_title="Emotion Detection App")

    st.title("Real-Time Emotion Detection")
    
    st.sidebar.title("Options")
    option = st.sidebar.selectbox(
        "Select media", 
        ("Webcam", "Image", "Video"), 
        index=None,
    )

    if option == "Webcam":
        if "detector" not in st.session_state:
            st.session_state.detector = EmotionDetector()

        detector = st.session_state.detector
        webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=lambda: detector)

        if webrtc_ctx.state.playing:
            detector.reset_emotion_times()  # Reset emotion times
            st.session_state.detector.last_update_time = time.time()  # Reset start time
            st.session_state.playing = True
        elif 'playing' in st.session_state and st.session_state.playing:
            st.session_state.playing = False
            emotion_times = detector.get_emotion_times()

            df_emotion_times = pd.DataFrame([emotion_times])
            st.write("Emotion Times DataFrame:")
            st.write(df_emotion_times.style)
            plot_emotion_pie_chart([emotion_times])
            
            st.session_state.all_emotion_times.append(emotion_times)
            df_all_emotion_times = pd.DataFrame(st.session_state.all_emotion_times)
            st.write("All Emotion Times DataFrame:")
            st.write(df_all_emotion_times.style)
            plot_emotion_pie_chart(st.session_state.all_emotion_times)
                
    elif option == "Image":
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
                emotion = emotion_dict[maxindex]
                
                prob = np.max(prediction)
                cv2.putText(img, f"{emotion} ({prob:.2f})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(img, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)

            # Convert the image to RGB (OpenCV uses BGR by default)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Convert the image to a PIL image
            im_pil = Image.fromarray(img_rgb)

            st.image(im_pil, use_column_width=True)
            

    elif option == "Video":
        video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(video_file.read())
            
            video_cap = cv2.VideoCapture(tfile.name)
            
            width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video_cap.get(cv2.CAP_PROP_FPS)
            
            fourcc = cv2.VideoWriter_fourcc(*'H264')  # Sử dụng codec H264
            out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
            
            detector = EmotionDetector()

            while video_cap.isOpened():
                ret, frame = video_cap.read()
                if not ret:
                    break

                processed_frame = detector.transformVideo(frame)
                out.write(processed_frame)

            video_cap.release()
            out.release()

            st.video('output.mp4')

            # In ra thời lượng cảm xúc của video
            emotion_times = detector.get_emotion_times()
            df_emotion_times = pd.DataFrame([emotion_times])
            st.write("Emotion Times DataFrame:")
            st.write(df_emotion_times.style)
            plot_emotion_pie_chart([emotion_times])
            
            st.session_state.all_emotion_times.append(emotion_times)
            df_all_emotion_times = pd.DataFrame(st.session_state.all_emotion_times)
            st.write("All Emotion Times DataFrame:")
            st.write(df_all_emotion_times.style)
            plot_emotion_pie_chart(st.session_state.all_emotion_times)

if __name__ == "__main__":
    main()





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

