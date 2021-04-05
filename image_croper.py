import cv2
import numpy as np
import time


def crop_img(in_img):
    if in_img.ndim > 2:
        sumimg = in_img.sum(2)  # sum color layers
    else:
        sumimg = in_img
    sumx = sumimg.sum(1)  # sum along x and y
    sumy = sumimg.sum(0)
    nonzero_x = np.asarray(np.nonzero(sumx))[0]
    nonzero_y = np.asarray(np.nonzero(sumy))[0]
    if len(nonzero_x) < 2:  # avoid zero
        nonzero_x = np.arange(1, 10, 1)
    if len(nonzero_y) < 2:
        nonzero_y = np.arange(1, 10, 1)
    max_x = nonzero_x.max()
    min_x = nonzero_x.min()
    max_y = nonzero_y.max()
    min_y = nonzero_y.min()
    width = max_x - min_x
    height = max_y - min_y
    if width > height:
        mid_y = int(0.5 * (max_y + min_y))
        half_width = int(0.5 * width)
        if half_width > mid_y:  # avoid by minus half_width the value become negtive
            in_img = cv2.copyMakeBorder(in_img, half_width, half_width, 0, 0
                                        , cv2.BORDER_CONSTANT, value=0)  # expand image in case we need more space
            img_croped = in_img[min_x:max_x, mid_y:mid_y + 2*half_width]
            crop_boundary = [min_x, max_x, mid_y, mid_y + 2*half_width]
        else:
            img_croped = in_img[min_x:max_x, mid_y - half_width:mid_y + half_width]
            crop_boundary = [min_x, max_x, mid_y - half_width, mid_y + half_width]
    else:
        mid_x = int(0.5 * (max_x + min_x))
        half_height = int(0.5 * height)
        if half_height > mid_x:
            in_img = cv2.copyMakeBorder(in_img, 0, 0, half_height, half_height
                                        , cv2.BORDER_CONSTANT, value=0)  # expand image in case we need more space
            img_croped = in_img[mid_x:mid_x + 2*half_height, min_y:max_y]
            crop_boundary = [mid_x, mid_x + 2*half_height, min_y, max_y]
        else:
            img_croped = in_img[mid_x - half_height:mid_x + half_height, min_y:max_y]
            crop_boundary = [mid_x - half_height, mid_x + half_height, min_y, max_y]
    return img_croped, crop_boundary


def effeciency_test():
    img = cv2.imread('data/randomg_resized/mask/mask0.png')
    t = time.time()
    sumimg = img.sum(2)  # 先累加三个颜色的图层(单色不用)
    print('sumlayer   ', time.time() - t)
    sumx = sumimg.sum(1)  # 沿x y轴合并
    print('sumx       ', time.time() - t)
    sumy = sumimg.sum(0)
    print('sumy       ', time.time() - t)
    nonzero_x = np.asarray(np.nonzero(sumx))
    print('nonzero_x  ', time.time() - t)
    nonzero_y = np.asarray(np.nonzero(sumy))
    print('nonzero_y  ', time.time() - t)
    max_x = nonzero_x.max()
    print('max_x      ', time.time() - t)
    min_x = nonzero_x.min()
    print('min_x      ', time.time() - t)
    max_y = nonzero_y.max()
    print('max_y      ', time.time() - t)
    min_y = nonzero_y.min()
    print('min_y      ', time.time() - t)
    k = 1
    for i in range(150):
        if sumx[i] > 0:
            k = sumx[i]
    print('for        ', time.time() - t)  # 这就是为什么不要用技术实践明显大于其他，其他几乎是瞬间完成


def crop_img_test():
    img = cv2.imread('data/classification/black/mask_with_image_crop/mask_with_image_crop0.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_crop, crop_boundary = crop_img(img)
    cv2.imshow('img', img)
    cv2.imshow('img_crop', img_crop)
    cv2.waitKey()


if __name__ == "__main__":
    crop_img_test()
