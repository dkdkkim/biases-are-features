import os
import sys

from tqdm import tqdm

import argparse

import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

"""
M. Jeon
2021-04-17
To explore which features are dedicated training ImageNet
"""

__ROOT_PATH = os.path.abspath('../../') 
sys.path.append(__ROOT_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help="resnet, alex, vgg, squeeze")
parser.add_argument('-batch_size', type=int, help="batch size")

args = parser.parse_args()
mode = args.mode
batch_size = args.batch_size

if mode == "resnet":
    model = models.resnet18(pretrained = True)
elif mode == "alex":
    model = models.alexnet(pretrained = True)
elif mode == "vgg":
    model = models.vgg16(pretrained = True)
else:   
    model = models.squeezenet1_0(pretrained = True)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
])
dataset = datasets.ImageNet("/data", split="val", transform = transform)
testloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

model.to(device)
model.eval()
total = 0.
correct = 0

with torch.no_grad():
    for data in tqdm(testloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"[ACUURACY:{mode}] {correct/len(dataset)*100}")
