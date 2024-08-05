import torch
import torch.nn as nn
import torch.quantization
from torchvision import transforms
from PIL import Image
import torchvision.datasets as datasets
import pickle

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample= None, stride= 1):
        super().__init__()

        self.expansion = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size= 3, stride= stride, padding= 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size= 3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace= True)
        self.identity_downsample = identity_downsample
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # x = self.relu(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        # x += identity
        x = self.skip_add.add(x, identity)
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, BasicBlock, layers, image_channels, num_classes):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size= 7, stride= 2, padding= 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace= True)
        self.maxpool = nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)

        self.layer1 = self._make_layer(BasicBlock, layers[0], out_channels= 64, stride= 1)
        self.layer2 = self._make_layer(BasicBlock, layers[1], out_channels= 128, stride= 2)
        self.layer3 = self._make_layer(BasicBlock, layers[2], out_channels= 256, stride= 2)
        self.layer4 = self._make_layer(BasicBlock, layers[3], out_channels= 512, stride= 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
    
        x = self.avgpool(x)
        x = self.dequant(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        

        return x


    def _make_layer(self, BasicBlock, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != out_channels:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels , kernel_size= 1, stride= stride),
                                               nn.BatchNorm2d(out_channels))
        layers.append(BasicBlock(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels

        for i in range(num_residual_blocks - 1):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

def ResNet18(img_channels= 3, num_classes= 10):
    return ResNet(BasicBlock, [2, 2, 2, 2], img_channels, num_classes)


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
model = torch.load('/home/tricus/task1/deeplearning-pytorch/res.pth')
# model = ResNet18()
model.train()
model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8 )
model.qconfig = torch.quantization.default_qconfig
model = torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)
model.load_state_dict(torch.load('/home/tricus/task1/deeplearning-pytorch/Resnet18QuantizedModel.pth', map_location='cpu'))


filename = 'testset_subset.pkl'
with open(filename, 'rb') as f:
    testset = pickle.load(f)
testset_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

img, label = testset[6]

model.eval()
img = transform(img)
img = img.unsqueeze(0) 


with torch.no_grad():
    output = model(img)
    _, predicted = torch.max(output, 1)
    print(f"Predicted class: {testset_classes[predicted.item()]},  Actual class: {testset_classes[label]}" )