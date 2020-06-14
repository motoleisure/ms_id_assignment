import numpy

###. Redis服务器地址、端口和数据库信息
HOST = "localhost"

REDIS_PORT = 6379
REDIS_DB = 0

## Mongo
MONGO_PORT = 27017
MONGO_DB = 'faceid_dbs'
MONGO_FACEID_EMD = 'faceid_emd'
MONGO_FACEID_IMGS = 'faceid_imgs'

###. 模型输入图片格式参数 
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 160
IMAGE_CHANS = 3
IMAGE_DTYPE = numpy.uint8

###. 队列参数
FACEID_REQ_IMAGE_QUEUE = "faceid_req_queue_000002"
BATCH_SIZE = 8

###. 睡眠参数
SERVER_SLEEP = 0.01
CLIENT_SLEEP = 0.01

###. 模型参数
GPU_FRACTION = 0.125
