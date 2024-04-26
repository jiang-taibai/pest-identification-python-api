import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from model import resnext50_32x4d
import torchvision.models.resnet  # 下载迁移学习权重


# 判断使用GPU训练或是使用CPU训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 对图像进行不同的转换处理，并组合
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 先随机采集，然后对裁剪得到的图像缩放为同一大小
                                 transforms.RandomHorizontalFlip(),  # 随机水平旋转给定图像
                                 transforms.ToTensor(),  # 转为Tensor
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  #  归一化处理
    "test": transforms.Compose([transforms.Resize(256),  # 照比例把图像最小的一个边长放缩到256，另一边按照相同比例放缩
                               transforms.CenterCrop(224),  # 中心裁剪
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # 获取根路径
image_path = data_root + "/data_set/bug_data/"  # 设置害虫图像路径
# 获得训练集数据和数量
train_dataset = datasets.ImageFolder(root=image_path+"train", transform=data_transform["train"])
train_num = len(train_dataset)

# 获得数据集索引
bug_list = train_dataset.class_to_idx
# 装换索引的key和val
cla_dict = dict((val, key) for key, val in bug_list.items())
# 将数据集索引转换为json对象
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# 载入训练集
batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# 载入测试集
test_dataset = datasets.ImageFolder(root=image_path + "test", transform=data_transform["test"])
val_num = len(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

resnet = resnext50_32x4d()  # 载入模型
model_weight_path = "ResNeXt50_32x4d.pth"  # 载入迁移学习权重
missing_keys, unexpected_keys = resnet.load_state_dict(torch.load(model_weight_path), strict=False)
# 重新赋值全连接层
in_channel = resnet.fc.in_features
resnet.fc = nn.Linear(in_channel, 22)
resnet.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.0001)
best_accurate = 0.0
save_path = 'ResNeXt50.pth'
for epoch in range(45):
    # 训练过程
    resnet.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logit = resnet(images.to(device))
        loss = loss_function(logit, labels.to(device))
        loss.backward()  # 反向传播
        optimizer.step()
        running_loss += loss.item()
        # 输出训练过程
        rate = (step+1)/len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.5f}".format(int(rate*100), a, b, loss), end="")
    print()
    # 验证过程
    resnet.eval()
    ac = 0.0
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data
            outputs = resnet(test_images.to(device))  # 输出最后一个全连接层
            predict_y = torch.max(outputs, dim=1)[1]
            ac += (predict_y == test_labels.to(device)).sum().item()
        test_accurate = ac / val_num
        if test_accurate > best_accurate:  # 保存最佳权重
            best_accurate = test_accurate
            torch.save(resnet.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' % (epoch + 1, running_loss / step, test_accurate))
print('训练结束')
