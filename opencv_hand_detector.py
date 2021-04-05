# https://github.com/spmallick/learnopencv/tree/master/HandPose
import cv2
import time
import numpy as np
from image_croper import crop_img


class HandDetector:
    def __init__(self, proto_file, weights_file, i_size):
        self.proto_file = proto_file
        self.weights_file = weights_file
        self.size = i_size
        self.nPoints = 22
        self.POSE_PAIRS = [
            [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11]
            , [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]
            , [5, 9], [9, 13], [13, 17], [2, 5], [6, 8], [10, 12], [14, 16], [18, 20]
        ]
        self.net = cv2.dnn.readNetFromCaffe(self.proto_file, self.weights_file)
        self.frameWidth = 0
        self.frameHeight = 0
        self.in_width = 0
        self.in_height = 0

    def load_vedio(self, in_cap, in_size):
        _, count_frame = in_cap.read()
        frame_use = cv2.resize(count_frame, in_size, interpolation=cv2.INTER_AREA)
        self.frameWidth = frame_use.shape[1]
        self.frameHeight = frame_use.shape[0]
        aspect_ratio = self.frameWidth / self.frameHeight
        self.in_height = 368
        self.in_width = int(((aspect_ratio * self.in_height) * 8) // 8)

    def network_update(self, in_frame):
        inp_blob = cv2.dnn.blobFromImage(in_frame, 1.0 / 255, (self.in_width, self.in_height), (0, 0, 0)
                                         , swapRB=False, crop=False)
        self.net.setInput(inp_blob)
        return self.net.forward()

    def selection(self, update_result, in_threshold):
        points = []
        for i in range(self.nPoints):
            # confidence map of corresponding body's part.
            probprob_map = update_result[0, i, :, :]
            probprob_map = cv2.resize(probprob_map, (self.frameWidth, self.frameHeight))

            # Find global maxima of the probprob_map.
            _, prob, _, point = cv2.minMaxLoc(probprob_map)

            if prob > in_threshold:
                # Add the point to the list if the probability is greater than the threshold
                points.append((int(point[0]), int(point[1])))
            else:
                points.append(None)
        return points

    def draw_skeleton(self, img, in_point):
        for pair in self.POSE_PAIRS:
            part_a = pair[0]
            part_b = pair[1]

            if in_point[part_a] and in_point[part_b]:
                cv2.line(img, in_point[part_a], in_point[part_b], (0, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(img, in_point[part_a], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(img, in_point[part_b], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    def draw_mask(self, in_point):
        img = np.zeros(self.size)
        more_line = self.POSE_PAIRS + [[1, 9], [2, 9], [1, 13], [1, 17], [2, 13], [2, 17]]
        linewide = 20
        for pair in more_line:
            part_a = pair[0]
            part_b = pair[1]
            if in_point[part_a] and in_point[part_b]:
                cv2.line(img, in_point[part_a], in_point[part_b], 255, linewide, lineType=cv2.LINE_AA)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(img, kernel)
        dilated = dilated.astype(np.uint8)
        return dilated, in_point[8]

    def hand_detection(self, in_threshold, img, switch='none'):
        output = self.network_update(img)
        # Empty list to store the detected keypoints
        point = self.selection(output, in_threshold)
        # Draw
        if switch == 'draw_skeleton':
            self.draw_skeleton(img, point)
        return self.draw_mask(point)


if __name__ == '__main__':
    protoFile = "model/hand/pose_deploy.prototxt"
    weightsFile = "model/hand/pose_iter_102000.caffemodel"
    input_source = "data/5orPalm.MP4"
    size = (500, 500)
    hand_detector = HandDetector(protoFile, weightsFile, size)
    # cap = cv2.VideoCapture(input_source)
    cap = cv2.VideoCapture(1)  # From web cam
    hand_detector.load_vedio(cap, size)

    k = 0
    while 1:
        k += 1
        t = time.time()
        hasFrame, frame = cap.read()
        frame_shrink = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frameCopy = np.copy(frame_shrink)
        if not hasFrame:
            cv2.waitKey()
            break
        threshold = 0.2
        mask, index_finger_tip = hand_detector.hand_detection(threshold, frame_shrink)
        img_with_mask = cv2.bitwise_and(frameCopy, frameCopy, mask=mask)
        crop_img_with_mask = crop_img(img_with_mask)
        print("Time Taken for frame = {}".format(time.time() - t))
        cv2.imshow('Output-Skeleton', frame_shrink)
        cv2.imshow('Mask', mask)
        cv2.imshow('img_with_mask', img_with_mask)
        cv2.imshow('crop_img_with_mask', crop_img_with_mask)

        if index_finger_tip:
            print(index_finger_tip)
        key = cv2.waitKey(10)
        if key == 27:
            break
