# Quantize_Pytorch_Vgg16AndMobileNet
Quantize,Pytorch,Vgg16,MobileNet

## Static Quantize And Aware training Quantize for VGG-16 and MobileNet-V2 

**Pytorch版本要求**：1.4+

**Note**：VGG-16中没有BN层，所以相较官方教程，去掉了fuse_model的融合部分

**项目目录结构**：

----Quantize_Pytorch：总项目文件夹

--------data：文件夹，存储imagenet_1k数据集

--------model：文件夹，存储VGG-16以及MobieNet的pretrained_model预训练float型模型

--------MobileNetV2-quantize_all.py：Static Quantize 和 QAT 两种方式

--------vgg16-static_quantize.py：Static Quantize to Vgg-16

--------vgg16-aware_train_quantize.py：QAT Quantize to Vgg-16

**数据集与预训练模型下载**：链接：https://pan.baidu.com/s/1-Rkrcg0R5DbNdjQ4umXBtQ ；提取码：p2um 
