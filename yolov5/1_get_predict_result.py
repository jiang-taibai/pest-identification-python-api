# ----------------------------------------------------------------------------------------------------------------------
# 检测图片
# ----------------------------------------------------------------------------------------------------------------------

from utils.datasets import *
from utils.utils import *
import argparse
import cv2
import os

# ----------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file')
parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
opt = parser.parse_args()
print(opt)


# ----------------------------------------------------------------------------------------------------------------------


class Yolo():
    def __init__(self):
        self.writer = None
        self.prepare()

    def prepare(self):
        global model, device, classes, colors, names
        device = torch_utils.select_device(device='0')

        google_utils.attempt_download(opt.weights)
        model = torch.load(opt.weights, map_location=device)['model'].float()

        model.to(device).eval()

        names = model.names if hasattr(model, 'names') else model.modules.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    def detect(self, frame):
        # im0 = imutils.resize(frame, width=720)
        im0 = frame
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

        for i, det in enumerate(pred):

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, score, cls in det:
                    label = '%s ' % (names[int(cls)])
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                    # ----------------------------------------------------------------------------------------------------------------------

                    predict_class = names[int(cls)]
                    boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(score), predict_class])

        return im0, boxes


# ----------------------------------------------------------------------------------------------------------------------

yolo = Yolo()
files = os.listdir('mAP/images/')
files.sort()

for file in files:
    if file.endswith('jpg') or file.endswith('png'):
        image_path = './mAP/images/' + file
        image = cv2.imread(image_path)

        try:
            _, boxes = yolo.detect(image)

            name = file.split('.')[0]
            writer = open('./mAP/input/detection-results/' + name + '.txt', 'a')

            for box in boxes:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                score = box[4]
                label = str(box[5])

                writer.write(label + ' '
                             + str(score)[:4] + ' '
                             + str(x1) + ' '
                             + str(y1) + ' '
                             + str(x2) + ' '
                             + str(y2) + '\n')

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow('', image)
            cv2.waitKey(3)

        except:
            print(image_path)

# ----------------------------------------------------------------------------------------------------------------------
