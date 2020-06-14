#*code utf-8*#
import redis
import json
import time
from pymongo import MongoClient
import numpy as np
import uuid

import sys
sys.path.append('../')
from facevi_utils import settings
from facevi_utils import helpers
from facevi.facemodel import FaceModel
from facevi.facedetector import FaceDetector
from facevi.faceutils import FaceUtils

# 设置gpu_memory_fraction

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)
KTF.set_session(sess)

def connect_to_redis():
    # connect to Redis server
    rdb = redis.StrictRedis(host=settings.HOST,
                           port=settings.REDIS_PORT, db=settings.REDIS_DB)
    return rdb

def connect_to_mongo(collection_name):
    # connect to MONGO server
    client = MongoClient(settings.HOST, settings.MONGO_PORT)
    # database : faceid_dbs
    mdb = client[settings.MONGO_DB]
    # collections : faceid_emd, faceid_imgs
    return mdb[collection_name]

# initializing face model
print("initializing face model ....................")
FD = FaceDetector(face_size_th=20)
FM = FaceModel()
FU = FaceUtils()

# def if_or_equal_faceid():
#     faceid_emd = connect_to_mongo('faceid_emd')
#     f_emds = faceid_emd.find()
#     faceid_imgs = connect_to_mongo('faceid_imgs')
#     f_imgs = faceid_imgs.find()
#
#     imgs_keys = []
#     for r1 in f_imgs:
#         imgs_keys.append(r1['face_id'])
#     for r2 in f_emds:
#         if r2['face_id'] not in imgs_keys:
#             return False
#     return True

## 从mongo中获得face_emd数据
## 数据结构 ： {_id : objectId, face_id : faceid, face_emd_2 : face_embedding}
## 数据结构 ： {_id : objectId, face_id : faceid, face_imgs : [face_img]}
def get_emds_from_mongo():
    faceid_emd = connect_to_mongo('faceid_emd')
    f_emds = faceid_emd.find()

    data_emds = {}
    for re in f_emds:
        data_emds[re['face_id']] = re['face_emd_2']
    return data_emds

def get_imgs_from_mongo():
    faceid_imgs = connect_to_mongo('faceid_imgs')
    f_imgs = faceid_imgs.find()

    data_imgs = {}
    for re in f_imgs:
        data_imgs[re['face_id']] = re['face_imgs']
    return data_imgs

## 更新existing face img list表
def update_existing_face_imgs_from_mongo(faceid, face_img):
    print('faceid ---------- {}'.format(faceid))
    faceid_imgs = connect_to_mongo('faceid_imgs')
    face_img_doc = faceid_imgs.find_one({'face_id':faceid})

    # print('face - img - list : {}'.format(face_img_doc))
    # print('face img doc : {}'.format(face_img_doc))
    image_list = face_img_doc['face_imgs']
    # image_list = helpers.b64decode_img_with_decompress(face_img_doc['face_imgs'])

    # print('image list type : {}'.format(type(image_list)))

    print('&^%$#@@@#    len image list : {}'.format(len(image_list)))
    if len(image_list) >= 10:
        image_list = image_list[1:]

    # import pdb
    # pdb.set_trace()

    image_list.append(helpers.b64encode_img_with_compress(face_img))

    new_data = {
        'face_id' : faceid,
        'face_imgs' : image_list
    }
    faceid_imgs.delete_one({'face_id': faceid})
    faceid_imgs.insert_one(new_data)

## 如果是新的face_id,往mongodb中插入新的记录
def insert_new_faceid_img(face_id, face_img):
    post_data = {
        'face_id' : face_id,
        'face_imgs' : [helpers.b64encode_img_with_compress(face_img)]
    }

    # print("insert new faceid img : {}".format(post_data))
    faceid_imgs = connect_to_mongo('faceid_imgs')
    faceid_imgs.insert_one(post_data)
    # import pdb
    # pdb.set_trace()

def insert_new_faceid_emd(face_id, face_emd):
    post_data = {
        'face_id' : face_id,
        'face_emd_2' : np.array(face_emd).tolist()
    }
    faceid_emd = connect_to_mongo('faceid_emd')
    faceid_emd.insert_one(post_data)

## 比较new_emd与数据库中的face_emd之间的距离，找到最小距离mini_dist的对应face_id
## 如果mini_dist比我们设定的阈值小，那么new_emd是数据库中的face_id，否则是一个新的face_id
def assign_faceid_service(new_emd, threash = 0.5):
    ### emd_redisdb : {faceid1 : emd1, faceid2 : emd2}
    emd_rdb = get_emds_from_mongo()
    # print("emd_redisdb---------------------")
    # print(type(emd_redisdb))
    # print(len(emd_redisdb))

    # img_rdb = get_imgs_from_mongo()
    # import pdb
    # pdb.set_trace()

    min_dist = 100
    index_key = None
    for key, emd in emd_rdb.items():
        dist = FM.distance(new_emd, np.asanyarray(emd))
        if dist < min_dist:
            min_dist = dist
            index_key = key

    if index_key is not None:
        print("min_dist : {}, threash : {}".format(min_dist, threash))
        if min_dist < threash:
            return index_key
    return -1

def assign_faceid_process_v3():
    rdb = connect_to_redis()
    # get the image from form
    # continually pool for new images to assign face id
    temp_index = 0
    while True:
        queue = rdb.lrange(settings.FACEID_REQ_IMAGE_QUEUE, 0, settings.BATCH_SIZE - 1)

        # loop over the queue
        """
            logits:
            1, get the image
            2, detect face
            3, tell face is aligned or not
            4, do face embedding
            5, compute the distances
            6, assign face id
            7, return key-value : {id, data {face_id, msg}}
        """
        """
            假设：一张图片最多包含一个人脸
            code解析：
            200 ： 成功获取图片的faceid，并且为已存在的faceid
            201 ： 成功获取图片的faceid，并且为新的faceid
            501 ： 获取图片faceid失败，图片没检测到人脸
            502 ： 获取图片faceid失败，人脸没有对齐(align)
        """
        for q in queue:
            print("-----------------new-----------------------")
            print("-----------------queue---------------------")
            print("-----------------new-----------------------")
            response = {}
            response['status'] = '501'
            response['data'] = {}
            # deserialize the object and obtain the input image
            q = json.loads(q.decode('utf-8'))
            id = q['id']
            # 1, get the image
            image = helpers.b64decode_img_with_decompress_with_shape(q['image'],
                        settings.IMAGE_DTYPE,
                        (settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, settings.IMAGE_CHANS))

            # 2, detect face
            face_scores, face_bbxes, face_keypoints = FD.detect_face(image)
            print('faces detected : {}'.format(len(face_bbxes)))
            if len(face_bbxes) == 0:
                response['data']['msg'] = 'no faces detected'
            # if len(face_keypoints) > 0:
            for fbb, fkp in zip(face_bbxes, face_keypoints):
                # 3, tell face is aligned or not
                is_align = FU.is_frontal_face(fkp)

                # 人脸对齐的
                if is_align:
                    print('image.shape : {}'.format(image.shape))
                    print('face bbx : {}'.format(fbb))
                    croped_face = image[fbb[1]:fbb[3], fbb[0]:fbb[2], :]
                    # 4, do face embedding
                    face_embedding = FM.inference([croped_face])

                    # flag = if_or_equal_faceid()
                    # print('flag : {}'.format(flag))

                    # 5, compute the distances and get the faceid
                    index_key = assign_faceid_service(face_embedding)
                    # 6, assign face id
                    # if index_key == -1, it's a new faceid
                    # else is an existing faceid

                    if index_key != -1:
                        print('existing faceid , index_key----{}'.format(index_key))
                        response['data']['msg'] = 'existing faceid'
                        response['data']['face_id'] = index_key
                        response['status'] = '200'

                        # temp_index = temp_index + 1
                        # if temp_index == 2:
                        #     import pdb
                        #     pdb.set_trace()
                        # update the face image for existing faceid
                        update_existing_face_imgs_from_mongo(index_key, croped_face)
                        # update_faceid(index_key, id, croped_face, face_embedding)
                    else:
                        response['data']['msg'] = 'new faceid'
                        response['data']['face_id'] = str(uuid.uuid4())
                        response['status'] = '201'
                        # insert new face image and face embedding
                        insert_new_faceid_img(response['data']['face_id'], croped_face)
                        insert_new_faceid_emd(response['data']['face_id'], face_embedding)
                else:
                    response['data']['msg'] = 'face is not aligned'
                    response['status'] = '502'

            rdb.set(id, json.dumps(response), ex=300)
            rdb.ltrim(settings.FACEID_REQ_IMAGE_QUEUE, 1, -1)

        time.sleep(settings.SERVER_SLEEP)

if __name__ == '__main__':
    # 处理双表
    assign_faceid_process_v3()
