# -*- coding: utf-8 -*-

from mtcnn.mtcnn import MTCNN
import cv2

cascade_file = 'facevi/haarcascade_frontalface_default.xml'

class FaceDetector():
    
    def __init__(self, detector_type="mtcnn", face_score_th=0.5, face_size_th=80):
        self.detector_type = detector_type
        self.face_score_th = face_score_th
        self.face_size_th = face_size_th
        self.face_detector = None
        if self.detector_type == "mtcnn":
            self._load_mtcnn_detector()
        else:
            self._load_opencv_detector()
            
    def detect_face(self, frame):
        if self.detector_type == "mtcnn":
            faces = self.face_detector.detect_faces(frame)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.face_size_th, self.face_size_th), flags=cv2.CASCADE_SCALE_IMAGE)
        
        face_scores, face_bbxes, face_keypoints = self._postprocess(faces, frame)
        
        return face_scores, face_bbxes, face_keypoints
        
    def _load_mtcnn_detector(self):
        self.face_detector = MTCNN()
    
    def _load_opencv_detector(self):
        self.face_detector = cv2.CascadeClassifier(cascade_file)
    
    def _postprocess(self, faces, frame):
        face_scores = []
        face_bbxes = []
        face_keypoints = []
        
        if self.detector_type == "mtcnn":
            for i in range(len(faces)):
                bounding_box = faces[i]['box']
                prob = faces[i]['confidence']
                keypoints = faces[i]['keypoints']
                if prob < self.face_score_th or bounding_box[2] < self.face_size_th \
                        or bounding_box[3] < self.face_size_th:
                    continue
                
                face_bbxes.append([bounding_box[0] if bounding_box[0] > 0 else 0,
                                   bounding_box[1] if bounding_box[1] > 0 else 0,
                bounding_box[0] + bounding_box[2] if bounding_box[0] + bounding_box[2] < frame.shape[0] else frame.shape[0],
                bounding_box[1] + bounding_box[3] if bounding_box[1] + bounding_box[3] < frame.shape[1] else frame.shape[1]])
                #
                # face_bbxes.append(
                #     [bounding_box[0] if bounding_box[0] > 0 else 0, bounding_box[1] if bounding_box[1] > 0 else 0,
                #      bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]])

                face_scores.append(prob)
                face_keypoints.append(keypoints)
        else:
            face_bbxes = [(y, x + w, y + h, x) for (x, y, w, h) in faces]
        
        return face_scores, face_bbxes, face_keypoints
    
