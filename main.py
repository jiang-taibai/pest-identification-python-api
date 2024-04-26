import json
import os
import sys
from tornado.web import Application, RequestHandler
from tornado.ioloop import IOLoop
from tornado.options import options
from urllib import parse
from resnet.predict import analyse_pest_singular
from yolov5.demo_image_main import analyse_pest_multiple

root_path = os.getcwd()
sys.path.append(root_path + "/resnet")
sys.path.append(root_path + "/yolov5")

port = 8899

options.define(
    name="port",
    default=port,
    type=int,
    help=None,
    metavar=None,
    multiple=False,
    group=None,
    callback=None
)


class BaseHandler(RequestHandler):
    """
    前后端分离,解决跨域问题
    """

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        # 这里要填写上请求带过来的Access-Control-Allow-Headers参数，如access_token就是我请求带过来的参数
        self.set_header("Access-Control-Allow-Headers", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS, DELETE")  # 请求允许的方法
        self.set_header("Content-Type", "application/json")  # 设置数据返回格式json
        self.set_header("Access-Control-Max-Age", "3600")  # 用来指定本次预检请求的有效期，单位为秒，，在此期间不用发出另一条预检请求。

    # vue 一般需要访问options方法， 如果报错则很难继续，所以只要通过就行了，当然需要其他逻辑就自己控制。
    def options(self):
        # 返回方法1
        self.set_status(204)
        self.finish()
        # 返回方法2
        self.write('{"errorCode":"00","errorMessage","success"}')


class IdentificationSingular(BaseHandler):

    def get(self):
        self.post()

    def post(self):
        # 获取post方式传递的参数
        img_data = self.get_argument("imgDataBase64")
        img_data = parse.unquote(img_data)
        # print(imgData)
        sys.path.insert(0, root_path + "/resnet")
        species_id, prob = analyse_pest_singular(img_data)
        res = {
            'speciesId': species_id,
            'prob': '{:.2f}'.format(prob)
        }
        self.write(json.dumps(res))


class IdentificationMultiple(BaseHandler):

    def get(self):
        self.post()

    def post(self):
        # 获取post方式传递的参数
        img_data = self.get_argument("imgDataBase64")
        img_data = parse.unquote(img_data)
        # print(imgData)
        sys.path.insert(0, root_path + "/yolov5")
        results, img_base64_data = analyse_pest_multiple(img_data)
        res = {
            'results': results,
            'imgBase64Data': img_base64_data
        }
        self.write(json.dumps(res))


if __name__ == "__main__":
    app = Application([
        (r"/analysePest/singular", IdentificationSingular),
        (r"/analysePest/multiple", IdentificationMultiple),
    ])
    print(f'正在监听：{port}端口')
    app.listen(port)
    IOLoop.current().start()
