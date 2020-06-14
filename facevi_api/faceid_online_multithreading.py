
import cv2
import threading
import numpy as np
from imutils.video import FPS

import sys
sys.path.append('../')
from facevi.facedetector import FaceDetector
from facevi_api.faceviAPI import UDC_API
from facevi_utils import helpers

# 设置gpu_memory_fraction

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)
KTF.set_session(sess)

def capture_face(margin = 15):
    vs = cv2.VideoCapture(0)
    # vs.set(3, 288)
    # vs.set(4, 512)
    # face detector model
    FD = FaceDetector(face_size_th=20)
    # localhost server
    appid = '6c21418d-436b-4231-a478-02dbbf706f4b'
    token = 'token2d437469-8be4-48b5-9cd6-42cc6cd99407'

    # nvidia server
    # appid = '7e8a6af2-4e8a-458b-9361-a02169412af3'
    # token = 'tokenbd355e4b-3d4b-475a-b349-a6fa5b8f2350'

    UDC_client = UDC_API(appid, token)
    # fps : computer the speed
    fps = FPS()
    fps.start()
    while True:
        res, frame = vs.read()
        fps.update()
        if not(res):
            break

        height, width, _ = frame.shape
        # frame = frame[96:384,64:576,:]
        print('[INFO] frame type : {}'.format(type(frame)))
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print('[INFO] image shape : {}'.format(frame.shape))
        # 1, detect faces
        face_scores, face_bbxes, face_keypoints = FD.detect_face(frame)
        print("[INFO] detect {} faces".format(len(face_bbxes)))

        for bbx in face_bbxes:
            # 2, crop face
            print("[INFO] bbx : {}".format(bbx))
            if bbx[0] > bbx[2] or bbx[1] > bbx[3]:
                continue
            # 给face图片加margin
            bbx[0] = 0 if bbx[0]-margin < 0 else bbx[0]-margin
            bbx[1] = 0 if bbx[1]-margin < 0 else bbx[1]-margin
            bbx[2] = width if bbx[2]+margin > width else bbx[2]+margin
            bbx[3] = height if bbx[3]+margin > height else bbx[3]+margin
            print("[INFO] bbx + margin : {}".format(bbx))

            croped_face = frame[bbx[1]:bbx[3], bbx[0]:bbx[2], :]
            # print("type(crop-face) : {}".format(type(croped_face)))
            # get faceid
            t = threading.Thread(target=UDC_client.get_faceid, args=[croped_face])

            # t = threading.Thread(target=UDC_client.get_faceid,
            #                      args=[helpers.b64encode_img_with_compress(croped_face)])

            # t = threading.Thread(target=assign_faceid_with_token, args=[croped_face])
            t.start()

            # rectangle for face
            cv2.rectangle(frame, (bbx[0],bbx[1]), (bbx[2],bbx[3]), (0,0,255), 1)
            print('------------segment----------------')

        cv2.imshow('test', frame)

        butt = cv2.waitKey(10) & 0xFF
        if butt == ord('q'):
            break

    cv2.destroyAllWindows()
    vs.release()
    fps.stop()

    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if __name__ == '__main__':
    capture_face()
