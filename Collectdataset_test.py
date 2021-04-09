import os
DIRECTORY = "F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_0"  # 这里是自己子文件夹的图片的位置，test_1到test_n
f = open('F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_0.txt', 'w')  # txt文件位置test_1到test_n
files = os.listdir(DIRECTORY)
print(files)
for file in files:
    print(file)
    f.writelines(file+" " + '0')  # num_class 是该类图像对应的分类一般用0-9
    f.write('\n')
f.close()

DIRECTORY = "F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_1"
f = open('F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_1.txt', 'w')
files = os.listdir(DIRECTORY)
print(files)
for file in files:
    print(file)
    f.writelines(file+" " + '1')
    f.write('\n')
f.close()

DIRECTORY = "F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_2"
f = open('F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_2.txt', 'w')
files = os.listdir(DIRECTORY)
print(files)
for file in files:
    print(file)
    f.writelines(file+" " + '2')
    f.write('\n')
f.close()

DIRECTORY = "F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_3"
f = open('F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_3.txt', 'w')
files = os.listdir(DIRECTORY)
print(files)
for file in files:
    print(file)
    f.writelines(file+" " + '3')
    f.write('\n')
f.close()

DIRECTORY = "F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_4"
f = open('F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_4.txt', 'w')
files = os.listdir(DIRECTORY)
print(files)
for file in files:
    print(file)
    f.writelines(file+" " + '4')
    f.write('\n')
f.close()

DIRECTORY = "F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_5"
f = open('F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_5.txt', 'w')
files = os.listdir(DIRECTORY)
print(files)
for file in files:
    print(file)
    f.writelines(file+" " + '5')
    f.write('\n')
f.close()

DIRECTORY = "F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_6"
f = open('F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_6.txt', 'w')
files = os.listdir(DIRECTORY)
print(files)
for file in files:
    print(file)
    f.writelines(file+" " + '6')
    f.write('\n')
f.close()

DIRECTORY = "F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_7"
f = open('F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_7.txt', 'w')
files = os.listdir(DIRECTORY)
print(files)
for file in files:
    print(file)
    f.writelines(file+" " + '7')
    f.write('\n')
f.close()

DIRECTORY = "F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_8"
f = open('F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_8.txt', 'w')
files = os.listdir(DIRECTORY)
print(files)
for file in files:
    print(file)
    f.writelines(file+" " + '8')
    f.write('\n')
f.close()

DIRECTORY = "F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_9"
f = open('F:\\McMaster courses\\789\\project\\Desfeng\\testdataset\\test_9.txt', 'w')
files = os.listdir(DIRECTORY)
print(files)
for file in files:
    print(file)
    f.writelines(file+" " + '9')
    f.write('\n')
f.close()
