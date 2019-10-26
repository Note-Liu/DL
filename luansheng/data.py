
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import itertools
import os
import pandas as pd
print(os.listdir("/home/liuxb/luan"))

#Checking if CUDA is available or not
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

df = pd.read_csv("/home/liuxb/luan/data/train_relationships.csv")
print(df.head())

'''
           p1          p2
0  F0002/MID1  F0002/MID3
1  F0002/MID2  F0002/MID3
2  F0005/MID1  F0005/MID2
3  F0005/MID3  F0005/MID2
4  F0009/MID1  F0009/MID4
'''

new = df["p1"].str.split("/", n = 1, expand = True)

# making separate first name column from new data frame 
df["Family1"]= new[0]
# making separate last name column from new data frame 
df["Person1"]= new[1]

# Dropping old Name columns
df.drop(columns =["p1"], inplace = True)

new = df["p2"].str.split("/", n = 1, expand = True)

# making separate first name column from new data frame 
df["Family2"]= new[0]
# making separate last name column from new data frame 
df["Person2"]= new[1]

# Dropping old Name columns
df.drop(columns =["p2"], inplace = True)
print(df.head())

'''
  Family1 Person1 Family2 Person2
0   F0002    MID1   F0002    MID3
1   F0002    MID2   F0002    MID3
2   F0005    MID1   F0005    MID2
3   F0005    MID3   F0005    MID2
4   F0009    MID1   F0009    MID4
'''

root_dir = '/home/liuxb/luan/data/train/'
temp = []
for index, row in df.iterrows():
    if os.path.exists(root_dir+row.Family1+'/'+row.Person1) and os.path.exists(root_dir+row.Family2+'/'+row.Person2):
        continue
    else:
        temp.append(index)
        
print(len(temp))                  #231
df = df.drop(temp, axis=0)


#A new column in the existing dataframe with all values as 1, since these people are all related
df['Related'] = 1

#Creating a dictionary, and storing members of each family
df_dict = {}
for index, row in df.iterrows():
    if row['Family1'] in df_dict:
        df_dict[row['Family1']].append(row['Person1'])
    else:
        df_dict[row['Family1']] = [row['Person1']]
        
#For each family in this dictionary, we'll first make pairs of people
#For each pair, we'll check if they're related in our existing Dataset
#If they're not in the dataframe, means we'll create a row with both persons and related value 0
i=1
for key in df_dict:
    pair = list(itertools.combinations(df_dict[key], 2))
    for item in pair:
        if len(df[(df['Family1']==key)&(df['Person1']==item[0])&(df['Person2']==item[1])])==0 \
        and len(df[(df['Family1']==key)&(df['Person1']==item[1])&(df['Person2']==item[0])])==0:
            new = {'Family1':key,'Person1':item[0],'Family2':key,'Person2':item[1],'Related':0}
            df=df.append(new,ignore_index=True)
        
#Storing rows only where Person1 and Person2 are not same
df = df[(df['Person1']!=df['Person2'])]

#len(df[(df['Related']==1)])

print(df['Related'].value_counts())

'''
1    3367
0    1566
'''

extra = df['Related'].value_counts()[1]-df['Related'].value_counts()[0]
while extra>=0:
    rows = df.sample(n=2)
    first = rows.iloc[0,:]
    second = rows.iloc[1,:]
    
    if first.Family1!=second.Family1 and first.Family2!=second.Family2:
        new1 = {'Family1':first.Family1,'Person1':first.Person1,'Family2':second.Family1,'Person2':second.Person1,'Related':0}
        extra=extra-1
        if extra==0:
            break
        new2 = {'Family1':first.Family2,'Person1':first.Person2,'Family2':second.Family2,'Person2':second.Person2,'Related':0}
        extra=extra-1
        
        df=df.append(new1,ignore_index=True)
        df=df.append(new2,ignore_index=True)


df = df.sample(frac=1).reset_index(drop=True)
print(df['Related'].value_counts())

'''
1    3367
0    3366
Name: Related, dtype: int64
'''
print(df.head())

'''
 Family1 Person1 Family2 Person2  Related
0   F0425    MID5   F0425    MID6        0
1   F0539    MID2   F0539    MID4        1
2   F0561    MID1   F0658    MID5        0
3   F0762    MID5   F0579    MID1        0
4   F0194    MID2   F0194    MID3        0
'''

