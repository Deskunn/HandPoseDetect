import random
import cv2
import matplotlib.pyplot as plt
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DIRECTORY = "F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test"  # 这里是自己的所有图片的位置
f = open('F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test.txt', 'r')

# 产生一个0-480的随机序列。480 是自己训练图片的总张数。用于将text.txt中的数据随机排序，训练时生成随机的batch
list = []
for i in range(0, 480):
    list.append(i)
random.shuffle(list)

##########分开train.txt中的数据###############
num = []  # labels数据集
imgs = []  # data数据
line = f.readline()
while line:
    a = line.split()  # 将txt分成两列
    c = a[0]+' '+a[1]
    data = c  # 这是选取图像的名称，一般是xxx.jpg或其他图片格式
    imgs.append(data)  # 将其添加在列表之中
    label = a[2]  # 这是选取图像的标签，一般是0-9的数字
    num.append(label)
    line = f.readline()
f.close()
print(imgs)

##############读取图片数据######################
batch = []  # 图像数据
labels = []  # 标签
for j in range(len(list)):  # 随机取出train文件夹中的图像
    num_1 = list[j]
    file_path = DIRECTORY+"\\"+imgs[num_1]  # 图像的位置
    print(file_path)
    img = cv2.imread(file_path)  # 将图像的信息读出来
    img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)  # 将图像变为指定大小
    batch.append(img)  # 图像数据存入batch中
    labels.append(num[num_1])  # 标签数据存入到labels中

batch_size = 32
batch_test = torch.Tensor(batch).permute(0, 3, 2, 1)/255
label_test = [int(x)for x in labels]
label_test = torch.Tensor(label_test).type(torch.LongTensor)
class_names = ['Maintaining', 'Powersaving', 'Scraming', 'Reporting', 'Resetting', 'Following', 'Startworking', 'Shutdown']
plt.figure(figsize=(20, 20))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(batch[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[label_test[i]])
plt.show()