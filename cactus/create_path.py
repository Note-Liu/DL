import os
import glob
dataset_dir = './data/raw_train/'     #原始训练集
train_dir = './data/train/'       #必须加"/"
valid_dir = './data/valid/'       #必须加"/"


if __name__ == "__main__":
    for root, dirs, files in os.walk(valid_dir):
        
        f1 = open("./data/valid_path.txt","w")
        for file in files:          
            f = open("./train.txt","r")
            for line in f.readlines():             
                if file in line:              
                    img_list2 = os.path.join(root+line)
                    f1.write(img_list2)
            f.close()
        f1.close()