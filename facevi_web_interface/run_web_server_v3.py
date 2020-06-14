# import the necessary packages

import numpy as np
import flask
import redis
import uuid
import time
import json
import io
from PIL import Image
import cv2

import sys
sys.path.append('../')
from facevi_utils import settings
from facevi_utils import helpers
from facevi_utils.auth import verify_appid

# initialize our Flask application and Redis server
app = flask.Flask(__name__)
rdb = redis.StrictRedis(host=settings.HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)

"""
    logits : 
    1, upload an image from browser 
    2, assign an id use uuid to form as {'id' : id, 'image' : image}
    3, push in request queue of redis database
    4, get response from redis by id
"""

@app.route("/get_faceid_by_token", methods=["POST"])
def get_faceid_by_token():
    rdb_len = rdb.llen(settings.FACEID_REQ_IMAGE_QUEUE)
    if rdb_len > 500:
        return flask.jsonify({'status':'504','data': {'msg':"too many requests error"}})
    data = flask.request.get_data().decode('utf8')
    appid = json.loads(data)['appid']
    token = json.loads(data)['token']
    print("appid : {}, token : {}".format(appid, token))
    ## appid和token验证
    res = verify_appid(appid, token)
    if res:
        print("token verified !--------")
        image = json.loads(data)['image']
        #  print("json image.shape {}".format(image.shape))
        # image = cv2.resize(helpers.b64decode_img_with_decompress(image), (160, 160), interpolation=cv2.INTER_AREA)
        image = helpers.b64decode_img_with_decompress(image, np.uint8)
        print("decompress image.shape {}".format(image.shape))

        image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)
        print("compress image.shape {}".format(image.shape))

        k = str(uuid.uuid4())
        d = {"id": k, "image": helpers.b64encode_img_with_compress(image)}
        rdb.rpush(settings.FACEID_REQ_IMAGE_QUEUE, json.dumps(d))

        # keep looping until our model server returns the output
        # predictions
        while True:
            # attempt to grab the output predictions
            output = rdb.get(k)

            # check to see if our model has classified the input
            # image
            if output is not None:
                # add the output predictions to our data
                # dictionary so we can return it to the client
                output = output.decode("utf-8")
                data = json.loads(output)

                # delete the result from the database and break
                # from the polling loop
                rdb.delete(k)
                break

            # sleep for a small amount to give the model a chance
            # to classify the input image
            time.sleep(settings.CLIENT_SLEEP)

        # return the data dictionary as a JSON response
        return flask.jsonify(data)
    else:
        return flask.jsonify({'status':'503','data': {'msg':"token testify error"}})

# for debugging purposes, it's helpful to start the Flask testing
# server (don't use this for production
if __name__ == "__main__":
    print("* Starting web service...")
    app.run(host='0.0.0.0')
