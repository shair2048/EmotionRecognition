import cv2
import mediapipe as mp
import time
import os
import csv

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDitectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDitectionCon
        self.minTrackCon = minTrackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode, 
                                                 max_num_faces=self.maxFaces, 
                                                 min_detection_confidence=self.minDetectionCon, 
                                                 min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, 
                                           self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    # print(x,y)
                    face.append([x,y])
                faces.append(face)
        return img, faces
    
def save_data(img, faces, output_file):
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for face in faces:
            writer.writerow([coord for landmark in face for coord in landmark])

def main(): 
    # cap = cv2.VideoCapture("videos/3.mp4")
    # pTime = 0
    detector = FaceMeshDetector()
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    
    # while True:
    #     success, img = cap.read()
        
    #     if not success:
    #         # print("Failed to read frame or video has ended!")
    #         break
        
    #     img, faces = detector.findFaceMesh(img)
    #     # if len(faces) != 0:
    #     #     print(len(faces))
            
    #     cTime = time.time()
    #     fps = 1/(cTime - pTime)
    #     pTime = cTime
        
    #     cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    #     cv2.imshow("Image", img)
    #     cv2.waitKey(1)
    
    for emotion in emotions:
        emotion_folder = f"images/train/{emotion}/"
        output_file = f"data/{emotion}_data.csv"
        
        # Xóa nội dung của file nếu file đã tồn tại
        if os.path.exists(output_file):
            os.remove(output_file)

        for filename in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
            
            img, faces = detector.findFaceMesh(img)
            if len(faces) != 0:
                save_data(img, faces, output_file)

    print("Data generation completed!")
    
if __name__ == "__main__":
    main()