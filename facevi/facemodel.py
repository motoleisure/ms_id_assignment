# -*- coding: utf-8 -*-

### 导入相关模块
import tensorflow as tf
from scipy import misc
import numpy as np
import math

class FaceModel(object):
    
    def __init__(self, model_file='../facevi/data/20180408-102900.pb', class_file=None, gpu_fraction=0.15, input_image_size=(160,160)):
        self.model_file = model_file
        self.class_file = class_file
        self.gpu_fraction = gpu_fraction
        self.input_image_size = input_image_size
        self.graph = None
        self.sess = None
        self.load_model()
    
    ### 加载模型
    # 过程：1. 设置gpu fraction, 2. 更新类的sess和graph属性
    def load_model(self):
        gpu_options = tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=self.gpu_fraction) 
        config = tf.ConfigProto(device_count={'gpu':0}, log_device_placement=False, allow_soft_placement=False, gpu_options=gpu_options)
        
        # load frozen tensorflow model into memory
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph, config=config)

        self.graph = detection_graph
        self.sess = sess
    
    ### 执行模型推理
    # 输入：图片列表，[img1, img2, img3, ...]， img on opencv BGR format.
    # 输出：特征列表, shape是(n,512)
    def inference(self, images):
        # Get input and output tensors
        images_placeholder = self.graph.get_tensor_by_name("input:0")
        embeddings = self.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")
        
        pre_images = self.preprocess(images)
        
        feed_dict = { images_placeholder: pre_images, phase_train_placeholder:False }
        embedding = self.sess.run(embeddings, feed_dict=feed_dict)
        
        return list(embedding)
    
    ### 图片预处理函数
    # 输入： 图片列表，[img1, img2, img3, ...]
    # 输出： numpy array, shape是 (n, height, width, channel)
    def preprocess(self, images):
        img_list = []
        for img in images:
            print('img shape : {}'.format(img.shape))
            p_img = misc.imresize(img, self.input_image_size, interp='bilinear')
            mean = np.mean(p_img)
            std = np.std(p_img)
            std_adj = np.maximum(std, 1.0/np.sqrt(p_img.size))
            p_img = np.multiply(np.subtract(p_img, mean), 1/std_adj)
    
            img_list.append(p_img)
            
        pre_images = np.stack(img_list)
        
        return pre_images
    
    @staticmethod
    def distance(embeddings1, embeddings2, distance_metric=0):
        if distance_metric==0:
            # Euclidian distance
            diff = np.subtract(embeddings1, embeddings2)
            dist = np.sqrt(np.sum(np.square(diff),1))
        elif distance_metric==1:
            # Distance based on cosine similarity
            dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
            norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
            similarity = dot / norm
            dist = np.arccos(similarity) / math.pi
        else:
            raise 'Undefined distance metric %d' % distance_metric 
            
        return dist