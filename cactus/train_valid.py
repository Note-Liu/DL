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
# import matplotlib.pyplot as plt
import torchvision.models as models
from tensorboardX import SummaryWriter

train_txt_path = "./data/train_path.txt"
valid_txt_path = "./data/valid_path.txt"


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, "r")
        imgs = []
        for line in fh:
            line = line.rstrip()  
            words = line.split()  
            imgs.append((words[0], int(words[1])))  
        self.imgs = imgs 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]  
        img = Image.open(fn).convert("RGB")  
        if self.transform is not None:
            img = self.transform(img)  
        return img, label

    def __len__(self):
        return len(self.imgs)


normMean = [0.4948052, 0.48568845, 0.44682974]
normStd = [0.24580306, 0.24236229, 0.2603115]
normTransform = transforms.Normalize(normMean, normStd)

data_transform = {"train": transforms.Compose([transforms.Resize(128),
                                               transforms.RandomCrop(64),
                                               transforms.ToTensor(),
                                               normTransform]),
                  "valid": transforms.Compose([transforms.Resize(64),
                                               transforms.ToTensor(),
                                               normTransform])
                  }

train_data = MyDataset(txt_path=train_txt_path, transform=data_transform["train"])
valid_data = MyDataset(txt_path=valid_txt_path, transform=data_transform["valid"])
image_datasets = {"train": train_data, "valid": valid_data}
dataset_sizes = {x:len(image_datasets[x]) for x in ["train","valid"]}
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=16)
dataloader = {"train": train_loader, "valid": valid_loader}

'''
inputs,labels = next(iter(dataloader["train"]))
print("X的个数{}".format(len(inputs)))
#example_classes = train_data.label
#print(example_classes)
def imshow(input,title = None):
    input = input.numpy().transpose((1,2,0))
    input = normStd*input+normMean
    input = np.clip(input,0,1)
    plt.imshow(input)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

inputs,labels = next(iter(dataloader["train"]))
print(labels)
out = torchvision.utils.make_grid(inputs)
imshow(out,title = [labels[x] for x in labels])
'''

print(torch.cuda.is_available())
Use_gpu = torch.cuda.is_available()

model = models.vgg16(pretrained = True)
for para in model.parameters():
    para.requires_grad = False
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088,4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096,4096),
                                       torch.nn.Sigmoid(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096,2))

print(model)
if Use_gpu:
    model = model.cuda()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0000001)
# optimizer = torch.optim.Adam(model.classifier.parameters(),lr = 0.0001)

writer = SummaryWriter('log')

epoch_n = 20
time_open = time.time()
for epoch in range(epoch_n):
    print("Epoch {}/{}".format(epoch, epoch_n - 1))
    print("-" * 10)
    for phase in ["train", "valid"]:
        if phase == "train":
            print("Training...")
            model.train(True)
        else:
            print("Validing...")
            model.train(False)
        running_loss = 0.0
        running_corrects = 0.0
        for batch, data in enumerate(dataloader[phase], 1):
            X, y = data

            y = torch.LongTensor(y)
            if Use_gpu:
                X, y = Variable(X.cuda()), Variable(y.cuda())
            else:
                X, y = Variable(X), Variable(y)
            y_pred = model(X)
            _, pred = torch.max(y_pred.data, 1)
            optimizer.zero_grad()
            y = y.long()
            loss = loss_fn(y_pred, y)
            if phase == "train":
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            running_corrects += torch.sum(pred == y.data.long())
            if batch % 200== 0 and phase == "train":
                print("Batch{},Train Loss:{:.4f},Train Acc:{:.4f}%".format(batch, running_loss / batch,
                                                                       100.0 * running_corrects / (16 * batch)))

        if phase == "train":
            epoch_loss = running_loss *16/dataset_sizes[phase]
            epoch_acc = 100*running_corrects.double()/dataset_sizes[phase]
            writer.add_scalar("Train/Loss",epoch_loss,epoch)
            writer.add_scalar("Train/Acc",epoch_acc,epoch)
        if phase == "valid":
            epoch_loss = running_loss*16/dataset_sizes[phase]
            epoch_acc = 100*running_corrects.double()/dataset_sizes[phase]
            writer.add_scalar("Test/Loss",epoch_loss,epoch)
            writer.add_scalar("Test/Acc",epoch_acc,epoch)


        epoch_loss = running_loss * 16 / len(image_datasets[phase])
        epoch_acc = 100 * running_corrects  / len(image_datasets[phase])
         
        print("{} Loss:{:.4f} Acc:{:.4f}%".format(phase, epoch_loss, epoch_acc))
writer.close()
time_end = time.time() - time_open
print(time_end)

torch.save(model,'./model/fullvgg16.pth')
torch.save(model.state_dict(), './model/simplevgg16.pth')