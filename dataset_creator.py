import cv2
import os
import numpy as np
import tensorflow as tf
from image_croper import crop_img


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")


def get_path():
    return os.getcwd()


def load_and_split(path, filename):
    main_path = get_path()
    mkdir(main_path + '\\data\\' + filename + '\\')
    cap = cv2.VideoCapture(path)
    is_opened = cap.isOpened()
    print(is_opened)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(fps, width, height)
    i = 0
    (flag, frame) = cap.read()
    while flag:
        '''
        if i <= 10:
            file_name = './data/' + 'image' + str(i) + '.jpg'
            i = i + 1
            if flag:
                cv2.imwrite(file_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        '''
        file_name = './data/' + filename + '/image' + str(i) + '.jpg'
        i = i + 1
        if flag:
            cv2.imwrite(file_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imshow('1', frame)
        (flag, frame) = cap.read()
        cv2.waitKey(10)
    ''''''
    print('end!')


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def hsl_mask(img, h, sensitivity, s_up, s_low, v_up, v_low):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_color = np.array([h - sensitivity, s_low, v_low])
    upper_color = np.array([h + sensitivity, s_up, v_up])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(mask, kernel1)
    eroded = cv2.erode(eroded, kernel2)
    # dilated = cv2.dilate(eroded, kernel1)
    return eroded


def img_with_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)


# test the parameters for mask
def color_test(image_path, h, sensitivity, s_up, s_low, v_up, v_low):
    # read image
    img = cv2.imread(image_path)
    img_show = resize_with_aspect_ratio(img, height=720)
    cv2.imshow('image', img_show)

    # add mask
    mask = hsl_mask(img, h, sensitivity, s_up, s_low, v_up, v_low)
    mask_with_image = img_with_mask(img, mask)
    mask_show = resize_with_aspect_ratio(mask, height=720)
    mask_with_image_show = resize_with_aspect_ratio(mask_with_image, height=720)
    cv2.imwrite('1.jpg', mask_show, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # show the output
    cv2.imshow('mask', mask_show)
    cv2.imshow('mask_with_image', mask_with_image_show)
    print(img.shape)
    print(mask.shape)

    # close the windows
    k = cv2.waitKey(0)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()


def path_manager(folder):
    main_path = get_path()
    path_image = main_path + '\\data\\' + folder + '\\image\\'
    # path_mask = main_path + '\\data\\' + folder + '\\mask\\'
    # path_mask_with_image = main_path + '\\data\\' + folder + '\\mask_with_image\\'
    path_mask_with_image_crop = main_path + '\\data\\' + folder + '\\mask_with_image_crop\\'

    mkdir(path_image)
    # mkdir(path_mask)
    # mkdir(path_mask_with_image)
    mkdir(path_mask_with_image_crop)
    # return path_image, path_mask, path_mask_with_image, path_mask_with_image_crop
    return path_image, path_mask_with_image_crop


def get_image_and_label_resize(path_of_video, folder, h, sensitivity, s_up, s_low, v_up, v_low, resize):
    # path_image, path_mask, path_mask_with_image, path_mask_with_image_crop = path_manager(folder)
    path_image, path_mask_with_image_crop = path_manager(folder)
    cap = cv2.VideoCapture(path_of_video)
    is_opened = cap.isOpened()
    print('The video can be open:', is_opened)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('The fps, width and height are ', fps, width, height)

    i = 0
    flag, frame = cap.read()
    while flag:
        image_file_name = path_image + 'image' + str(i) + '.jpg'
        # mask_file_name = path_mask + 'mask' + str(i) + '.png'
        # mask_with_image_file_name = path_mask_with_image + 'mask_with_image' + str(i) + '.jpg'
        mask_with_image_crop_file_name = path_mask_with_image_crop + 'mask_with_image_crop' + str(i) + '.jpg'
        i += 1

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame_gray = cv2.resize(frame_gray, resize, interpolation=cv2.INTER_AREA)
        mask = hsl_mask(frame, h, sensitivity, s_up, s_low, v_up, v_low)
        mask = cv2.resize(mask, resize, interpolation=cv2.INTER_AREA)
        mask_with_image = img_with_mask(frame_gray, mask)
        mask_with_image_crop = crop_img(mask_with_image)
        mask_with_image_crop = cv2.resize(mask_with_image_crop, resize, interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(image_file_name, frame_gray, [cv2.IMWRITE_JPEG_QUALITY, 100])
        # cv2.imwrite(mask_file_name, mask, [cv2.IMWRITE_PNG_BILEVEL, 0])
        # cv2.imwrite(mask_with_image_file_name, mask_with_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(mask_with_image_crop_file_name, mask_with_image_crop, [cv2.IMWRITE_JPEG_QUALITY, 100])

        cv2.imshow('mask_with_image_resize', mask_with_image)
        cv2.imshow('frame_gray_resize', frame_gray)
        cv2.imshow('mask_resize', mask)
        cv2.imshow('mask_with_image_crop', mask_with_image_crop)
        (flag, frame) = cap.read()
        cv2.waitKey(2)
    print('end!')


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image_example(image_string, mask_string):
    image_shape = tf.image.decode_jpeg(image_string).shape
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'image': _bytes_feature(image_string),
        'mask': _bytes_feature(mask_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def size_counter(image_path):
    total_files = 0
    for _, _, files in os.walk(image_path):
        for Files in files:
            total_files += 1
    return total_files


def create_tfrecords(image_path, mask_path, file_name):
    files_number = size_counter(image_path)
    with tf.io.TFRecordWriter(file_name) as writer:
        for k in range(files_number):
            image_file_name = image_path + 'image' + str(k) + '.jpg'
            mask_file_name = mask_path + 'mask' + str(k) + '.jpg'
            k += 1
            image_string = open(image_file_name, 'rb').read()
            mask_string = open(mask_file_name, 'rb').read()
            tf_example = image_example(image_string, mask_string)
            writer.write(tf_example.SerializeToString())


if __name__ == "__main__":
    # Don't forget to delete the old folder
    '''
    filenames = ['1', '2', '3', '4', '5orPalm',
                 'closePalm', 'down', 'fist', 'fistSide', 'ok', 'point', 'thumb',
                 '1g', '2g', '3g', '4g', '5orPalmg',
                 'closePalmg', 'downg', 'fistg', 'fistSideg', 'okg', 'pointg', 'thumbg',
                 'randomg']
    for i in filenames:
        load_and_split("./data/" + i + ".MP4", i)
    '''
    # color_test('./data/1g/image0.jpg', 60, 20, 255, 43, 255, 46)
    '''
    filenames = ['1g', '2g', '3g', '4g', '5orPalmg',
                 'closePalmg', 'downg', 'fistg', 'fistSideg', 'okg', 'pointg', 'thumbg',
                 'randomg']
    '''
    ''''''
    # filenames = ['1orPoint', '2', '3', '4', '5orPalm', 'down', 'fist', 'fistMove', 'ok', 'palmMove', 'point']
    filenames = ['black']
    for i in filenames:
        get_image_and_label_resize("./data/MP4/" + i + ".MP4", i, 60, 20, 255, 43, 255, 46, (227, 227))

    # print(size_counter('./data_set/image/'))


    '''
    filenames = ['1g', '2g', '3g', '4g', '5orPalmg',
                 'closePalmg', 'downg', 'fistg', 'fistSideg', 'okg', 'pointg', 'thumbg',
                 'randomg']
    for i in filenames:
        create_tfrecords('./data/'+i+'/image/', './data/'+i+'/mask/', './data/data_'+i+'.tfrecords')
    '''

