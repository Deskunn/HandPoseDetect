import random
import cv2
import matplotlib.pyplot as plt
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DIRECTORY = "F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test"
f = open('F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test.txt', 'r')

list = []
for i in range(0, 480):
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
    file_path = DIRECTORY+"\\"+imgs[num_1]
    print(file_path)
    img = cv2.imread(file_path)
    img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)
    batch.append(img)
    labels.append(num[num_1])

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