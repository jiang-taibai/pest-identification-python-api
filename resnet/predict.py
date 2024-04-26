import torch
import json
from .model import resnext50_32x4d
from PIL import Image
from torchvision import transforms
import re
import base64
from io import BytesIO


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


def analyse_pest_singular(img_data):
    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = base64_to_image(img_data)
    img = data_transform(img)
    # 对图像进行解压
    img = torch.unsqueeze(img, dim=0)
    try:
        json_file = open('resnet/class_indices.json', 'r')
        class_log = json.load(json_file)
    except Exception as ex:
        print(ex)
        exit(-1)
    # 载入模型和权重
    model = resnext50_32x4d(num_classes=22)
    model_weight_path = "resnet/ResNeXt50.pth"
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()
    dic = {"1草履蚧成虫": "C15331005010", "2麻皮蝽成虫": "C15408105005", "3日本脊吉丁": "C21301095005", "4星天牛": "C21701065010",
           "5桑天牛成虫": "C21701080005", "6松墨天牛": "C21701445005", "7柳蓝叶甲": "C21703280010", "8黄刺蛾": "C22301070005",
           "8黄刺蛾幼虫": "C22301070005", "9褐边绿刺蛾幼虫": "C22301090015", "9褐边绿刺蛾": "C22301090015", "10霜天蛾": "C22341165010",
           "10霜天蛾幼虫": "C22341165010", "11杨扇舟蛾": "C22342015005", "11杨扇舟蛾幼虫": "C22342015005", "12杨小舟蛾": "C22342135005",
           "13美国白蛾": "C22345105005", "13美国白蛾幼虫": "C22345105005", "14人纹污灯蛾": "C22345170045", "14人纹污灯蛾幼虫": "C22345170045",
           "15丝带凤蝶": "C22359050005", "15丝带凤蝶幼虫": "C22359050005"}
    with torch.no_grad():
        # 对数据的维度进行压缩
        output = torch.squeeze(model(img))
        # 获得概率分布
        predict = torch.softmax(output, dim=0)
        # 获取概率最大处所对应的索引值
        predict_class = torch.argmax(predict).numpy()
    # print(class_log[str(predict_class)], dic[class_log[str(predict_class)]], predict[predict_class].numpy())
    return dic[class_log[str(predict_class)]], predict[predict_class].float()
