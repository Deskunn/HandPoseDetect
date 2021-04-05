import cv2
import numpy as np
import time
from opencv_hand_detector import HandDetector
from image_croper import crop_img
from tensorflow.keras.models import load_model
import tensorflow as tf

protoFile = "model/hand/pose_deploy.prototxt"
weightsFile = "model/hand/pose_iter_102000.caffemodel"
# input_source = "data/MP4/demo/Control.MP4"
input_source = "data/MP4/demo/IMG_4787.MP4"
model = load_model('model/classification/classification.h5')
size = (500, 500)
hand_detector = HandDetector(protoFile, weightsFile, size)
cap = cv2.VideoCapture(input_source)
# cap = cv2.VideoCapture(1)  # From web cam
hasFrame, frame = cap.read()
hand_detector.load_vedio(cap, size)
mode1 = 'number'
mode2 = 'command'
mode1_class_names = ['1', '2', '4', '5', 'None', 'None', '0', '0', '3', '0']
mode2_class_names = ['point', 'start', 'shift', 'stop', 'none', 'hold', 'ready', 'ready', 'confirm', 'ready']
# class_names = ['1orPoint', 'two', 'four', '5orPalm', 'None', 'down', 'fist', 'fist', 'ThreeOrConfirm', 'fist']
mode = mode1  # default mode
class_names = mode1_class_names
font = cv2.FONT_HERSHEY_COMPLEX
vid_writer = cv2.VideoWriter(mode+'.avi',
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (frame.shape[1], frame.shape[0]))

k = 0
while 1:
    k += 1
    t = time.time()
    hasFrame, frame = cap.read()
    # A copy for final display
    frame_copy = np.copy(frame)
    if not hasFrame:
        cv2.waitKey()
        break
    frame_shape = frame.shape
    frame_shrink = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    # find hand
    threshold = 0.2
    # mask, index_finger_tip = hand_detector.hand_detection(threshold, frame_shrink, 'draw_skeleton')
    mask, index_finger_tip = hand_detector.hand_detection(threshold, frame_shrink)
    # data preparation
    img_with_mask = cv2.bitwise_and(frame_shrink, frame_shrink, mask=mask)
    crop_img_with_mask, crop_boundary = crop_img(img_with_mask)
    crop_img_with_mask_gray = cv2.cvtColor(crop_img_with_mask, cv2.COLOR_RGB2GRAY)
    crop_img_with_mask_gray = np.stack((crop_img_with_mask_gray,) * 3, axis=-1)
    crop_img_with_mask_gray = cv2.resize(crop_img_with_mask_gray, (227, 227), interpolation=cv2.INTER_AREA)
    crop_img_with_mask_gray = tf.expand_dims(crop_img_with_mask_gray, 0)
    # prediction
    prediction = model(crop_img_with_mask_gray)
    predicted_label = np.argmax(prediction)

    # output
    # draw
    img_with_label = frame_copy
    rate = (frame_shape[0]/size[0], frame_shape[1]/size[1])
    if crop_boundary[1]-crop_boundary[0] > 10:  # null crop filter
        cv2.rectangle(img_with_label, (int(crop_boundary[3] * rate[1]), int(crop_boundary[1] * rate[0])),
                      (int(crop_boundary[2] * rate[1]), int(crop_boundary[0] * rate[0])),
                      (255, 255, 255), thickness=3)
    label = 'Gesture: ' + class_names[predicted_label]
    cv2.putText(img_with_label, label, (0, 45), font, 2, (255, 255, 255), 2)
    label = 'Mode: ' + mode
    cv2.putText(img_with_label, label, (0, frame_shape[0]-10), font, 2, (255, 255, 255), 2)
    if index_finger_tip and class_names[predicted_label] == 'point':
        label = 'fingertip position is ' + str(index_finger_tip)
        cv2.putText(img_with_label, label, (0, 90), font, 1, (255, 255, 255), 2)

    # img
    # cv2.imshow('frame_shrink', frame_shrink)
    # cv2.imshow('Output-Skeleton', frame_shrink) # only when hand_detection has the third parameter
    # cv2.imshow('img_with_mask', img_with_mask)
    cv2.imshow('crop_img_with_mask', crop_img_with_mask)
    cv2.imshow('img_with_label', img_with_label)
    # print('Gesture is ', class_names[predicted_label])
    # print('possibility ', 100 * np.max(prediction))
    # print("Time Taken for frame = {}".format(time.time() - t))
    # videoWrite
    vid_writer.write(img_with_label)
    key = cv2.waitKey(10)
    if key == 49:  # key 1
        mode = mode1
        class_names = mode1_class_names
    if key == 50:  # key 2
        mode = mode2
        class_names = mode2_class_names
    if key == 27:
        break

vid_writer.release()