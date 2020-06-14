# import the necessary packages
import numpy as np
import base64
import sys
import cv2


def get_max_bbx(bbx_list):
    # input : [(x1,y1, w1, h1), (x2,y2,w2,h2),....]
    # get max (w1 * h1)
    max_ = 0
    index = 0
    for i in range(len(bbx_list)):
        area = bbx_list[i][2] * bbx_list[i][3]
        if max_ < area:
            max_ = area
            index = i
    return bbx_list[index]


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


def b64decode_img_with_decompress(a):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    # convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.b64decode(a), dtype=np.uint8)
    a = cv2.imdecode(a, cv2.IMREAD_COLOR)

    # return the decoded image
    return a
