import redis
import uuid
import json, time

import sys
sys.path.append('../')
from facevi_utils import settings

# connect to Redis server
rdb = redis.StrictRedis(host=settings.HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)

def generate_appid_token():
    for i in range(1):
        appid = {'appid' : str(uuid.uuid4())}
        token = 'token' + str(uuid.uuid4())
        # exp - 1 day
        rdb.set(token, json.dumps(appid))
        # rdb.set(token, json.dumps(appid), ex=86400)
        print('appid : {}, token : {}'.format(appid.get('appid'), token))
        time.sleep(0.01)

if __name__ == '__main__':
    generate_appid_token()