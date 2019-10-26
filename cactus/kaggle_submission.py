import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from typing import Any
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import time
import matplotlib.pyplot as plt
import torchvision.models as models

test_path = '/kaggle/input/aerial-cactus-identification/test/test'
test_txt_path = "/kaggle/input/test-path/test_path.txt"


# test_path = '../input/test/test/'
class MyDataset(Dataset):

    def __init__(self, labels, data_dir, transform=None):
        super().__init__()
        self.labels = labels.values
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        name, label = self.labels[index]
        img_path = os.path.join(self.data_dir, name)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)


normMean = [0.50371784, 0.45253035, 0.4685982]
normStd = [0.1518169, 0.14036989, 0.15396805]
normTransform = transforms.Normalize(normMean, normStd)
data_transform = {"train": transforms.Compose([transforms.Resize(256),
                                               transforms.RandomCrop(224),
                                               transforms.ToTensor(),
                                               normTransform]),
                  "valid": transforms.Compose([transforms.Resize(224),
                                               transforms.ToTensor(),
                                               normTransform])
                  }

'''
#显示一个batchsize图片
inputs,labels = next(iter(dataloader))
img = torchvision.utils.make_grid(inputs)
img = img.numpy().transpose(1,2,0)
img = img *normStd+normMean
img = np.clip(img,0,1)    #将[0,1]变成[0,255]
plt.imshow(img)
plt.show()
'''

def write_csv(results, file_name):
    import csv
    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "has_cactus"])
        writer.writerows(results)


print(torch.cuda.is_available())

model = models.vgg16(pretrained=False)
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.Sigmoid(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2))
model.load_state_dict(torch.load('/kaggle/input/cactusvgg16/cactusvgg16.pth'))

# print(model)
Use_gpu = torch.cuda.is_available()
if Use_gpu:
    model = model.cuda()

sub = pd.read_csv("/kaggle/input/aerial-cactus-identification/sample_submission.csv")

sub.info()
test_data = MyDataset(labels=sub, data_dir=test_path, transform=data_transform["valid"])
dataloader = DataLoader(dataset=test_data, batch_size=16, shuffle=False)
results = []

for batch, data in enumerate(dataloader):
    X, path = data
    if Use_gpu:
        X = Variable(X.cuda())
    else:
        X = Variable(X)
    y_pred = model(X)  
    probability = torch.nn.functional.softmax(y_pred, dim=1)[:, 1].data.tolist()
    for i in probability:
        results.append(i)

sub['has_cactus'] = results
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("END")
