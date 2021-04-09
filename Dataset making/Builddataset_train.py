import random
import cv2
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_files(file_dir, f):
    list = []
    for i in range(0, 1920):
        list.append(i)
    random.shuffle(list)

    num = []
    imgs = []
    line = f.readline()
    while line:
        a = line.split()
        c = a[0]+' '+a[1]
        data = c
        imgs.append(data)
        label = a[2]
        num.append(label)
        line = f.readline()
    f.close()
    print(imgs)


    batch = []
    labels = []
    for j in range(len(list)):
        num_1 = list[j]
        file_path = file_dir +"\\"+imgs[num_1]
        print(file_path)
        img = cv2.imread(file_path)
        img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)
        batch.append(img)
        labels.append(num[num_1])

    batch_size = 32
    batch_train = torch.Tensor(batch).permute(0, 3, 2, 1)/255
    label_train = [int(x)for x in labels]
    label_train = torch.Tensor(label_train).type(torch.LongTensor)

    return batch_train, label_train

