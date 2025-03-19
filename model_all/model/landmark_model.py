import dlib
import cv2
import numpy as np
import mediapipe as mp

class dilb_landmark():
    def __init__(self):
        # 使用 Dlib 的正面人脸检测器 frontal_face_detector
        self.detector = dlib.get_frontal_face_detector()
        # Dlib 的 68点模型
        self.predictor = dlib.shape_predictor("../pretrained_model/shape_predictor_68_face_landmarks.dat")
    def get_landmark(self,image_path):
        # 读取图片
        img = cv2.imread(image_path)
        # 将图片转换为灰度图（Dlib 通常在灰度图像上表现更好）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        faces = self.detector(gray, 1)
        shape = self.predictor(gray,faces[0])
        landmark=[]
        if shape:
            for j in range(68):  # 68个关键点
                x = shape.part(j).x
                y = shape.part(j).y
                landmark.append((x,y))
        else:
            print(f"{image_path} is fault")
            return False
        landmark=np.asanyarray(landmark)
        return landmark
class mediapipe_landmark():
    def __init__(self):
        # 初始化 MediaPipe 面部检测器
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
    def get_landmark(self,image_path):
        # 加载图片
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 面部特征点检测
        with self.mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            results = face_mesh.process(image_rgb)
        landmarks=[]
        # 如果检测到面部
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x = np.float16(landmark.x * image.shape[1])
                    y = np.float16(landmark.y * image.shape[0])
                    landmarks.append((x,y))
        else:
            print(f"{image_path} is fault")
            return False
        landmarks=np.asanyarray(landmarks)
        return landmarks   