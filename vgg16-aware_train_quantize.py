"""
==========================
    Style: aware training quantize (QAT)
    Model: VGG-16
    Create by: Han_yz @ 2020/1/29
    Email: 20125169@bjtu.edu.cn
    Github: https://github.com/Forggtensky
==========================
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
import sys
import torch.quantization

"""
------------------------------
    1、Model architecture
------------------------------
"""

class VGG(nn.Module):
    def __init__(self,features,num_classes=1000,init_weights=False):
        super(VGG,self).__init__()
        self.features = features  # 提取特征部分的网络，也为Sequential格式
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(  # 分类部分的网络
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes)
        )
        # add the quantize part
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,start_dim=1)
        # x = x.mean([2, 3])
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module,nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias,0)
            elif isinstance(module,nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(module.bias,0)

cfgs = {
    'vgg11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'vgg19':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M'],
}

def make_features(cfg:list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]  #vgg采用的池化层均为2,2参数
        else:
            conv2d = nn.Conv2d(in_channels,v,kernel_size=3,padding=1)  #vgg卷积层采用的卷积核均为3,1参数
            layers += [conv2d,nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)  #非关键字的形式输入网络的参数

def vgg(model_name='vgg16',**kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg),**kwargs)  # **kwargs为可变长度字典，保存多个输入参数
    return model


"""
------------------------------
    2、Helper functions
------------------------------
"""

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5


def run_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed

def load_model(model_file):
    model_name = "vgg16"
    model = vgg(model_name=model_name,num_classes=1000,init_weights=False)
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


"""
------------------------------
    3. Define dataset and data loaders
------------------------------
"""

def prepare_data_loaders(data_path):
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    print("dataset_train : %d" % (len(dataset)))

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    print("dataset_test : %d" % (len(dataset_test)))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test

"""
------------------------------
    4. Output un-quantize model
------------------------------
"""

# Specify random seed for repeatable results
torch.manual_seed(191009)

data_path = 'data/imagenet_1k'
saved_model_dir = 'model/'
float_model_file = 'vgg16_pretrained_float.pth'
scripted_float_model_file = 'vgg16_quantization_scripted.pth'
qat_quantized_model_file = 'vgg16_qat_model.pth'

train_batch_size = 30
eval_batch_size = 30

data_loader, data_loader_test = prepare_data_loaders(data_path)
criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir + float_model_file).to('cpu')

print('\n Before quantization: \n',float_model)
float_model.eval()

# Note: vgg-16 has no BN layer so that not need to fuse model

num_eval_batches = 10

print("Size of baseline model")
print_size_of_model(float_model)

# to get a “baseline” accuracy, see the accuracy of our un-quantized model
top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
# torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file) # save un_quantized model


"""
------------------------------
    5. QAT quantize part
------------------------------
"""

def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train() # must be in train mode
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return


qat_model = load_model(saved_model_dir + float_model_file)
# qat_model.fuse_model()

qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

torch.quantization.prepare_qat(qat_model, inplace=True)
print('\n\nAfter preparation for QAT, look the fake-quantization modules: \n',qat_model)

optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
num_train_batches = 20
num_epochs = 4

# Train and check accuracy after each epoch
for nepoch in range(num_epochs):
    train_one_epoch(qat_model, criterion, optimizer, data_loader, torch.device('cpu'), num_train_batches)
    if nepoch > 3:
        # Freeze quantize parameters
        qat_model.apply(torch.quantization.disable_observer)
    if nepoch > 2:
        # Freeze batch norm mean and variance estimates
        qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # Check the accuracy after each epoch
    quantized_model = torch.quantization.convert(qat_model.eval(), inplace=False)
    quantized_model.eval()
    top1, top5 = evaluate(quantized_model,criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Epoch %d :Evaluation accuracy on %d images, %2.2f'%(nepoch, num_eval_batches * eval_batch_size, top1.avg))

torch.jit.save(torch.jit.script(quantized_model), saved_model_dir + qat_quantized_model_file) # save qat_quantized model

def run_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed

print("\ninference time compare")
run_benchmark(saved_model_dir + scripted_float_model_file, data_loader_test)
run_benchmark(saved_model_dir + qat_quantized_model_file, data_loader_test)

