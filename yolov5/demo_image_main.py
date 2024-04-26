# ----------------------------------------------------------------------------------------------------------------------
# 检测图片
# ----------------------------------------------------------------------------------------------------------------------
from .utils import google_utils
from .utils.datasets import *
from .utils.utils import *
import argparse
import imutils
import cv2
import re
import base64
from io import BytesIO

# ----------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='yolov5/weights/best.pt', help='path to weights file')
parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
opt = parser.parse_args()
print(opt)


labels = ['C15331005010',
          'C15408105005',
          'C21301095005',
          'C21701065010',
          'C21701080005',
          'C21701445005',
          'C21701445005_2',
          'C21703280010',
          'C21703280010_2',
          'C22301070005',
          'C22301070005_2',
          'C22301090015',
          'C22301090015_2',
          'C22341165010',
          'C22341165010_2',
          'C22342015005',
          'C22342015005_2',
          'C22342135005',
          'C22342135005_2',
          'C22345105005',
          'C22345105005_2',
          'C22345170045',
          'C22345170045_2',
          'C22359050005',
          'C22359050005_2']

# ----------------------------------------------------------------------------------------------------------------------


class Yolo():
    def __init__(self):
        self.writer = None
        self.prepare()

    def prepare(self):
        global model, device, classes, colors, names
        device = torch_utils.select_device(device='cpu')

        google_utils.attempt_download(opt.weights)
        model = torch.load(opt.weights, map_location=device)['model'].float()

        model.to(device).eval()

        names = model.names if hasattr(model, 'names') else model.modules.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    def detect(self, frame):
        im0 = imutils.resize(frame, width=720)
        img = letterbox(frame, new_shape=416)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

        boxes = []
        confidences = []
        classIDs = []

        nums = [0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0]
        for i, det in enumerate(pred):

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, score, cls in det:
                    label = '%s ' % (names[int(cls)]) + str(float(score))[:4]
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                    # ----------------------------------------------------------------------------------------------------------------------

                    boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])])
                    confidences.append(float(score))
                    classIDs.append(int(cls))

                    idx = labels.index(names[int(cls)])
                    nums[idx] += 1

            return im0, nums


# ----------------------------------------------------------------------------------------------------------------------

def base64_to_image(base64_str, image_path=None):
    base64_str = re.sub('^data:image/.+;base64,', '', base64_str)
    missing_padding = 3 - len(base64_str) % 3
    base64_str += '=' * missing_padding
    byte_data = base64.b64decode(base64_str)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    if image_path:
        img.save(image_path)
    return img


def analyse_pest_multiple(base64_str):
    yolo = Yolo()
    input_path = 'yolov5/inference/input.jpg'
    output_path = 'yolov5/out/out.jpg'

    # 保存图片至 inputPath
    base64_to_image(base64_str, input_path)
    # cv2读取图片
    image = cv2.imread(input_path)
    # 预测获得数据
    image, nums = yolo.detect(image)
    # 获得结果
    results = []
    for i in range(0, 25):
        if nums[i] > 0:
            results.append({
                'pestId': str(labels[i])[:12],
                'quantity': str(nums[i])
            })
    # 保存结果图片
    cv2.imwrite(output_path, image)
    # 再读取图片
    with open(output_path, 'rb') as f:
        base64_data = base64.b64encode(f.read())
        imgBase64Data = base64_data.decode()
    return results, imgBase64Data

# ----------------------------------------------------------------------------------------------------------------------
