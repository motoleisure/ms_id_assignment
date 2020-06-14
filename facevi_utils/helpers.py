# import the necessary packages
import numpy as np
import base64
import sys
import cv2


def base64_encode_image(a):
    # base64 encode the input NumPy array
    return base64.b64encode(a).decode("utf-8")


def base64_decode_image(a, dtype, shape):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    # convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)

    # return the decoded image
    return a


def b64encode_img_with_compress(a):
    return base64_encode_image(cv2.imencode('.jpg', a)[1])


def b64decode_img_with_decompress(a, dtype):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    # convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.b64decode(a), dtype=dtype)
    a = cv2.imdecode(a, cv2.IMREAD_COLOR)

    # return the decoded image
    return a

def b64decode_img_with_decompress_with_shape(a, dtype, shape):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    # convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.b64decode(a), dtype=dtype)
    a = cv2.imdecode(a, cv2.IMREAD_COLOR)
    a = a.reshape(shape)

    # return the decoded image
    return a

