import os
import glob

test_dir = '/content/drive/cactus/data/test/'

if __name__ == "__main__":
    for root, dirs, files in os.walk(test_dir):
        f1 = open("/content/drive/cactus/data/test_path.txt", "w")
        for file in files:
            img_list = os.path.join(root + file + '\n')  
            f1.write(img_list)
        f1.close()
