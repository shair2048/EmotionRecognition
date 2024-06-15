import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, VideoHTMLAttributes
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
model = tf.keras.models.load_model('best_model.keras')
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

# Global array to store emotion times from each session
if "all_emotion_times" not in st.session_state:
    st.session_state.all_emotion_times = []

# Define a video transformer class for emotion detection
class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.emotion_times = {emotion: 0 for emotion in emotion_dict.values()}
        self.last_emotions = {}
        self.last_update_time = time.time()
        self.lock = threading.Lock()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
    def process_frame(self, frame, show_probabilities=False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        current_time = time.time()
        with self.lock:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray[np.newaxis, :, :, np.newaxis] / 255.0  # Normalize to [0, 1]

                prediction = model.predict(roi_gray)
                maxindex = int(np.argmax(prediction))
                emotion = emotion_dict[maxindex]

                # Update time for current emotion
                self.emotion_times[emotion] += current_time - self.last_update_time

                prob = np.max(prediction)
                cv2.putText(frame, f"{emotion} ({prob:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                if show_probabilities:
                    data = {emotion: [round(prediction[0][i], 2)] for i, emotion in enumerate(emotion_dict.values())}
                    df = pd.DataFrame(data)
                    st.write(df)

            self.last_update_time = current_time

        return frame

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = self.process_frame(img)
        return img

    def transformVideo(self, frame):
        frame = self.process_frame(frame)
        return frame
    
    def transformImg(self, image_file):
        show_probabilities = True
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = self.process_frame(img, show_probabilities)

        # Convert the image to RGB (OpenCV uses BGR by default)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the image to a PIL image
        im_pil = Image.fromarray(img_rgb)

        # Display the image in Streamlit
        st.image(im_pil, use_column_width=True)
        
    def reset_emotion_times(self):
        with self.lock:
            self.emotion_times = {emotion: 0 for emotion in emotion_dict.values()}
            self.last_emotion = None
            self.last_update_time = time.time()

    def get_emotion_times(self):
        with self.lock:
            current_time = time.time()
            if self.last_emotions:
                for emotion in self.last_emotions.values():
                    self.emotion_times[emotion] += current_time - self.last_update_time
            self.last_update_time = current_time
            
        # Format the values of emotion_times to have 2 decimal places
        formatted_emotion_times = {emotion: f"{time_spent:.2f}" for emotion, time_spent in self.emotion_times.items()}
        
        return formatted_emotion_times


def plot_emotion_pie_chart(emotion_history):
    total_times = {emotion: 0 for emotion in emotion_dict.values()}
    for session in emotion_history:
        for emotion, time in session.items():
            total_times[emotion] += float(time) 

    # Filter out emotions with zero time
    total_times = {emotion: time for emotion, time in total_times.items() if time > 0}
    # Plot the pie chart
    fig, ax = plt.subplots(facecolor='none')
    ax.pie(total_times.values(), startangle=90)
    ax.axis('equal') 

    fig.set_size_inches(3, 3)

    total_sum = sum(total_times.values())
    percentages = [f"{key} ({value / total_sum * 100:.1f}%)" for key, value in total_times.items()]
    plt.legend(percentages, loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small')

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
                detector = EmotionDetector() 
                detector.transformImg(image_file)

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
