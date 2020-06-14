
from urllib import request
import json
import cv2, base64

class UDC_API:
    def __init__(self, appid, token):
        self.appid = appid
        self.token = token
        # self.__api_url = 'http://192.168.3.6:5000'
        self.__api_url = 'http://localhost:5000'
        self.__action_face_id = 'get_faceid_by_token'

        # self.update_token()

    def __base64_encode_image(self, a):
        # base64 encode the input NumPy array
        return base64.b64encode(a).decode("utf-8")

    def __b64encode_img_with_compress(self, a):
        return self.__base64_encode_image(cv2.imencode('.jpg', a)[1])

    def update_token(self):
        pass

    def __compress_and_send_request(self, full_request_url, data_dict):
        data = json.dumps(data_dict).encode(encoding='utf-8')
        header_dict = {"Content-Type": "application/json"}
        req = request.Request(url=full_request_url, data=data, headers=header_dict)
        res = request.urlopen(req)
        res = res.read()
        return res.decode(encoding='utf-8')

    # Input : np.ndarray, bgr
    # Input : base64 str

    def get_faceid(self, image_np_bgr):
        full_request_url = '/'.join([self.__api_url, self.__action_face_id])
        data = {
            'appid': self.appid,
            'token': self.token
        }
        if type(image_np_bgr) == type('str'):
            data['image'] = image_np_bgr
        else:
            data['image'] = self.__b64encode_img_with_compress(image_np_bgr)

        res = self.__compress_and_send_request(full_request_url, data)
        # print('res : {}'.format(res))
        return res

    def update_image_by_faceid(self, faceid, face_img):
        pass

    def __age_gender_predict(self):
        pass

    def get_age_gender(self):
        pass


        