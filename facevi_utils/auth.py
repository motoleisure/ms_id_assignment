import json
import redis
from facevi_utils import settings

# connect to Redis server
rdb = redis.StrictRedis(host=settings.HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)

def verify_appid(appid, token):
    _appid = json.loads(rdb.get(token).decode('utf8')).get('appid')

    # print('verifying appid : {}'.format(_appid))
    if appid == _appid:
        return True
    else:
        return False

