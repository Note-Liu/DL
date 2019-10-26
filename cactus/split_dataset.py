#coding:utf-8


import torch
import os
import glob
import random
import shutil

dataset_dir = 'drive/cactus/data/raw_train/'     #原始训练集
train_dir = 'drive/cactus/data/train/'      
valid_dir = 'drive/cactus/data/valid/'       
train_per = 0.8
valid_per = 0.2

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
if __name__ == "__main__":
    for root,dirs,files in os.walk(dataset_dir):
        for sDir in dirs:                       
            imgs_list = glob.glob(os.path.join(root,sDir) + '/*.jpg')  
            random.seed(666)
            random.shuffle(imgs_list)                  
            imgs_num = len(imgs_list)                    
            train_point = int(imgs_num * train_per)       
            #print(train_point)
            valid_point = int(imgs_num * valid_per)   
            #print(valid_point)
            for i in range(imgs_num):
                if  i < train_point:                  
                    out_dir = train_dir                 
                else:
                    out_dir = valid_dir

                makedir(out_dir)
                out_path = out_dir 
                shutil.copy(imgs_list[i],out_path)  
            print("train:{},valid:{}".format(train_point,imgs_num-train_point))