# 1. 环境准备

使用 Anaconda 创建一个基于 3.8 的虚拟环境

```shell
conda create -n pest_identify python=3.8
```

激活虚拟环境

```shell
conda activate pest_identify
```

安装依赖（注意一定要先激活虚拟环境，否则会安装到别的环境里去了）

```shell
pip3 install -r requirements.txt
```

如果是在无GPU虚拟机中运行，需要安装 CPU 版本的 PyTorch

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

# 2. 权重下载

- 夸克网盘：[https://pan.quark.cn/s/5c16857e48c1](https://pan.quark.cn/s/5c16857e48c1) 提取码：UVAe
- 联系方式：如果你需要其他方式下载，请联系我的邮箱 `emailtojiang@gmail.com`

将 `ResNeXt50.pth` 和 `ResNeXt50_32x4d.pth` 放到 `resnet/` 目录下
将 `YOLOv5.pt` 放到 `yolov5/weights/` 目录下

# 3. 执行

```shell
python main.py
```

# 4. 相关配置

- 端口：`main.py` 中的 `port` 变量，默认 `8899`