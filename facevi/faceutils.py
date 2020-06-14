# -*- coding: utf-8 -*-
import numpy as np
from math import isnan

'''
def align_object(p1, p2):
    center_point = (int((p1[0] + p2[0])/2), int((p1[1] + p2[1])/2))
    center_r = math.sqrt((p1[0] - p2[0]) ** 2+(p1[1] - p2[1]) ** 2)
    tan_value = (p2[1] - p1[1])/(p2[0] - p1[0])
    angle = (math.atan(tan_value) * 180)/math.pi
    
    return center_point, int(float(center_r/2)), angle

def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated

#angle, center_point, center_r, half_width)
def align_and_crop_face(image, face_bbxes, face_keypoints):
    ## Input: image, face_bbxes, face_keypoints
    ## Ouput: face_img list
    
    rotate_img = rotate(image, (-1)*angle, center_point)
    #cv2.imshow("test", rotate_img)
    xmin = center_point[0] - half_width if center_point[0] - half_width >= 0 else 0
    ymin = center_point[1] - center_r if center_point[1] - center_r >= 0 else 0
    xmax = center_point[0] + half_width if center_point[0] + half_width <= rotate_img.shape[1] else rotate_img.shape[1]
    ymax = center_point[1] + center_r if center_point[1] + center_r <= rotate_img.shape[0] else rotate_img.shape[0]
    return rotate_img[ymin:ymax, xmin:xmax, :], xmin, ymin, xmax, ymax
'''

def area_triangle(pts):
    ## pts = [p1,p2,p3], p1 = [x1,y1]
    d12 = np.sqrt(np.sum(np.square(np.subtract(pts[0],pts[1]))))
    d23 = np.sqrt(np.sum(np.square(np.subtract(pts[1],pts[2]))))
    d32 = np.sqrt(np.sum(np.square(np.subtract(pts[2],pts[0]))))
    
    s = (d12 + d23 + d32) / 2
    area = (s*(s-d12)*(s-d23)*(s-d32)) ** 0.5
    
    if isnan(area):
        print("area is nan")
        area = 0
    else:
        area = int(area)
        
    return area

class FaceUtils():
    
    ### Input: 5 face keypoints
    ### Output: True: Frontal Face, False: non frontal
    @staticmethod
    def is_frontal_face(face_keypoints):
        up_triangle = np.array([face_keypoints['left_eye'],face_keypoints['right_eye'],face_keypoints['nose']])
        left_triangle = np.array([face_keypoints['left_eye'],face_keypoints['nose'],face_keypoints['mouth_left']])
        right_triangle = np.array([face_keypoints['right_eye'],face_keypoints['nose'],face_keypoints['mouth_right']])
        down_triangle = np.array([face_keypoints['mouth_left'],face_keypoints['nose'],face_keypoints['mouth_right']])
        
        left_area = area_triangle(left_triangle)
        right_area = area_triangle(right_triangle)
        up_area = area_triangle(up_triangle)
        down_area = area_triangle(down_triangle)
        
        ## LR_COND
        if right_area == 0:
            LR_COND = False
        else:
            LR = abs((left_area - right_area) / right_area)
            LR_COND = True if LR < 0.6 else False
        
        ## UD_COND
        if down_area == 0:
            UD_COND = False
        else:
            UD = (up_area - down_area) / down_area
            UD_COND = True if -0.1 < UD  and UD < 1.8 else False
        
        return LR_COND and UD_COND
    
    ### Input: 
    ###   1. frame: rgb frame
    ###   2. face_bbxes, face_keypoints: face detector output
    ### Output:
    ###   face_img_List: can be used as the input of FaceModel to get the face embedding.
    def align_and_crop_face(frame, face_bbxes, face_keypoints):
        face_img_list = []
        return face_img_list

