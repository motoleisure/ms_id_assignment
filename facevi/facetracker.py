# -*- coding: utf-8 -*-
from mtcnn.mtcnn import MTCNN
import numpy as np

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    ### START CODE HERE ### (≈ 5 lines)
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(yi2 - yi1, 0) * max(xi2 - xi1, 0)
    ### END CODE HERE ### 

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ### START CODE HERE ### (≈ 3 lines)
    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
    union_area = box1_area + box2_area - inter_area
    ### END CODE HERE ###
    
    # compute the IoU
    ### START CODE HERE ### (≈ 1 line)
    iou = inter_area / union_area
    ### END CODE HERE ###
    
    return iou

class Face():
    def __init__(self, face_id, face_bbx, face_ntime):
        self.face_id = face_id
        self.face_bbx = face_bbx
        self.face_ntime = face_ntime
        self.face_bbx_list = [face_bbx]
        self.face_ntime_list = [face_ntime]
    
    def update(self, face_bbx, face_ntime):
        self.face_bbx = face_bbx
        self.face_ntime = face_ntime
        self.face_bbx_list.append(face_bbx)
        self.face_ntime_list.append(face_ntime)

class FaceTracker():
    def __init__(self, face_detector=MTCNN(), face_score_th=0.6, face_size_th=80):
        self.face_detector = face_detector
        self.face_list = []                                # 当前画面所有处于激活状态的Face对象
        self.face_timeout_counter = []                     # 跟踪当前所有face对象，如果counter超过指定阈值，则删除当前Face对象
        self.face_ntime = ""
        self.face_bbxes = []
        self.face_scores = []
        self.face_keypoints = []
        self.face_score_th = face_score_th
        self.face_size_th = face_size_th                   # 如果face_size_th 大于 1，表示绝对值，小于1表示百分比; 分别长 和 宽进行比较，而不是计算面积
        self._face_counts = 0
    
    def detect(self, frame, frame_ntime):
        self.face_ntime = frame_ntime
        faces = self.face_detector.detect_faces(frame)
        print(len(faces))
        self._postprocess(faces)
        
    def update(self):
        ### 遍历所有当前所有的detected faces, 判断属于哪个face，否则新建一个face
        for one_face_bbx in self.face_bbxes:
            
            # 计算 iou
            face_iou_list = []
            for face_obj in self.face_list:
                face_iou_list.append(iou(one_face_bbx, face_obj.face_bbx))
            
            # 判断属于哪个face对象
            if len(face_iou_list) > 0:
                matched_face_id = np.argmax(np.array(face_iou_list))
                print(face_iou_list[matched_face_id])
                if face_iou_list[matched_face_id] > 0.4:
                    self.face_list[matched_face_id].update(one_face_bbx, self.face_ntime)
                    self.face_timeout_counter[matched_face_id] = 0
                    continue
            
            # 没有找到匹配的face对象， 新建一个face对象
            self._face_counts += 1
            self.face_list.append(Face(self._face_counts, one_face_bbx, self.face_ntime))
            self.face_timeout_counter.append(0)
            
        ### 删除所有超时没有更新的face对象
        new_face_list = []
        new_face_timeout_counter = []
        for face_index in range(len(self.face_list)):
            if self.face_timeout_counter[face_index] > 6:
                continue
            else:
                new_face_list.append(self.face_list[face_index])
                new_face_timeout_counter.append(self.face_timeout_counter[face_index] + 1)
        
        self.face_list = new_face_list
        self.face_timeout_counter = new_face_timeout_counter
        
    def _postprocess(self, faces):
        self.face_bbxes = []
        self.face_scores = []
        self.face_keypoints = []
        for i in range(len(faces)):
            bounding_box = faces[i]['box']
            prob = faces[i]['confidence']
            keypoints = faces[i]['keypoints']
            if prob < self.face_score_th or bounding_box[2] < self.face_size_th or bounding_box[3] < self.face_size_th:
                continue
    
            bbx = (bounding_box[0], bounding_box[1], bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3])        
            self.face_bbxes.append(bbx)
            self.face_scores.append(prob)
            self.face_keypoints.append(keypoints)
            
    # -*- coding: utf-8 -*-
from mtcnn.mtcnn import MTCNN
import numpy as np

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    ### START CODE HERE ### (≈ 5 lines)
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(yi2 - yi1, 0) * max(xi2 - xi1, 0)
    ### END CODE HERE ### 

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ### START CODE HERE ### (≈ 3 lines)
    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
    union_area = box1_area + box2_area - inter_area
    ### END CODE HERE ###
    
    # compute the IoU
    ### START CODE HERE ### (≈ 1 line)
    iou = inter_area / union_area
    ### END CODE HERE ###
    
    return iou

class Face():
    def __init__(self, face_id, face_bbx, face_ntime):
        self.face_id = face_id
        self.face_bbx = face_bbx
        self.face_ntime = face_ntime
        self.face_bbx_list = [face_bbx]
        self.face_ntime_list = [face_ntime]
    
    def update(self, face_bbx, face_ntime):
        self.face_bbx = face_bbx
        self.face_ntime = face_ntime
        self.face_bbx_list.append(face_bbx)
        self.face_ntime_list.append(face_ntime)

class FaceTracker():
    def __init__(self, face_detector=MTCNN(), face_score_th=0.6, face_size_th=80):
        self.face_detector = face_detector
        self.face_list = []                                # 当前画面所有处于激活状态的Face对象
        self.face_timeout_counter = []                     # 跟踪当前所有face对象，如果counter超过指定阈值，则删除当前Face对象
        self.face_ntime = ""
        self.face_bbxes = []
        self.face_scores = []
        self.face_keypoints = []
        self.face_score_th = face_score_th
        self.face_size_th = face_size_th                   # 如果face_size_th 大于 1，表示绝对值，小于1表示百分比; 分别长 和 宽进行比较，而不是计算面积
        self._face_counts = 0
    
    def detect(self, frame, frame_ntime):
        self.face_ntime = frame_ntime
        faces = self.face_detector.detect_faces(frame)
        print(len(faces))
        self._postprocess(faces)
        
    def update(self):
        ### 遍历所有当前所有的detected faces, 判断属于哪个face，否则新建一个face
        for one_face_bbx in self.face_bbxes:
            
            # 计算 iou
            face_iou_list = []
            for face_obj in self.face_list:
                face_iou_list.append(iou(one_face_bbx, face_obj.face_bbx))
            
            # 判断属于哪个face对象
            if len(face_iou_list) > 0:
                matched_face_id = np.argmax(np.array(face_iou_list))
                print("face_iou_list[matched_face_id] : {}".format(face_iou_list[matched_face_id]))
                if face_iou_list[matched_face_id] > 0.4:
                    self.face_list[matched_face_id].update(one_face_bbx, self.face_ntime)
                    self.face_timeout_counter[matched_face_id] = 0
                    continue
            
            # 没有找到匹配的face对象， 新建一个face对象
            self._face_counts += 1
            self.face_list.append(Face(self._face_counts, one_face_bbx, self.face_ntime))
            self.face_timeout_counter.append(0)
            
        ### 删除所有超时没有更新的face对象
        new_face_list = []
        new_face_timeout_counter = []
        for face_index in range(len(self.face_list)):
            if self.face_timeout_counter[face_index] > 6:
                continue
            else:
                new_face_list.append(self.face_list[face_index])
                new_face_timeout_counter.append(self.face_timeout_counter[face_index] + 1)
        
        self.face_list = new_face_list
        self.face_timeout_counter = new_face_timeout_counter
        
    def _postprocess(self, faces):
        self.face_bbxes = []
        self.face_scores = []
        self.face_keypoints = []
        for i in range(len(faces)):
            bounding_box = faces[i]['box']
            prob = faces[i]['confidence']
            keypoints = faces[i]['keypoints']
            if prob < self.face_score_th or bounding_box[2] < self.face_size_th or bounding_box[3] < self.face_size_th:
                continue
    
            bbx = (bounding_box[0], bounding_box[1], bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3])        
            self.face_bbxes.append(bbx)
            self.face_scores.append(prob)
            self.face_keypoints.append(keypoints)
            
    
