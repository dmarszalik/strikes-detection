"""
# Fighters pose detection
The goal of this part of project is to detect body points XY coordinates for fighters.

source:
https://learnopencv.com/yolo-nas-pose/
"""

import torch
# source: https://docs.deci.ai/super-gradients/latest/documentation/source/models.html
from super_gradients.training import models
import cv2
import numpy as np
# https://docs.ultralytics.com/
from ultralytics import YOLO

"""
keypoints_mapping = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle',
}
The origin (0,0) is at the top-left corner of the image
"""

"""
I've recorded myself to train model whether fighter is knocked-down or not.
"""


class PoseDetector:
    def __init__(self, model_name, pretrained_weights="coco_pose"):
        self.model = models.get(model_name, pretrained_weights=pretrained_weights)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # checks if GPU is available
        self.model.to(self.device) # if ('cuda'), the model will be transferred to the GPU, otherwise to the CPU.

    def detect_pose(self, image, confidence=0.51):
        """ Function that returns XY coordinates of bboxes and bodypoints XY coordinates """
        preds = self.model.predict(image, conf=confidence)
        prediction = preds[0].prediction
        return prediction.bboxes_xyxy, prediction.poses


class FighterDetector:
    def __init__(self, yolo_model_path):
        self.yolo_model = YOLO(yolo_model_path)

    def is_fighter(self, image_box):
        """ Function is taking image box, checks if shows fighter and returns boolean value """

        if image_box.shape[0] == 0 or image_box.shape[1] == 0:
            return False
        else:
            yolo_results = self.yolo_model(image_box)
            filtered_fighters = [box for box in yolo_results[-1].boxes.data if box[-1] == 0]
            filtered_fighters.sort(key=lambda x: x[-2], reverse=True)
            filtered_referee = [box for box in yolo_results[-1].boxes.data if box[-1] == 2]
            filtered_referee.sort(key=lambda x: x[-2], reverse=True)

            if len(filtered_fighters) > 0 and len(filtered_referee) > 0:
                chance_fighter = filtered_fighters[0][-2]
                chance_referee = filtered_referee[0][-2]
            elif len(filtered_fighters) > 0 and len(filtered_referee) == 0:
                try:
                    chance_fighter = filtered_fighters[0][-2]
                finally:
                    print(filtered_fighters[0])
                chance_referee = 0
            else:
                chance_fighter = 0
                chance_referee = 0

            return chance_fighter > chance_referee


class VideoProcessor:
    def __init__(self, pose_detector, fighter_detector, take_every=10, limit=None):
        self.pose_detector = pose_detector
        self.fighter_detector = fighter_detector
        self.take_every = take_every
        self.limit = limit

    def process_video(self, video_path):
        poses_data_list = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return None

        current_frame = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                print("End of video file.")
                break

            if current_frame % self.take_every == 0:
                poses = self.detect_fighter_poses(frame)
                poses_data_list.extend(poses)

            current_frame += 1
            if isinstance(self.limit, int) and current_frame >= self.limit:
                print("Maximum number of frames reached.")
                break

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        poses_data_np = np.array(poses_data_list)
        return poses_data_np

    def detect_fighter_poses(self, frame):
        poses_data = []
        if frame.shape[0] > 0 and frame.shape[1] > 0:
            bboxes, poses = self.pose_detector.detect_pose(frame)
            for i in range(len(bboxes)):
                if bboxes.shape[0] > 0:
                    x1, y1, x2, y2 = map(int, bboxes[i])
                    cropped_image = frame[y1:y2, x1:x2]
                    if self.fighter_detector.is_fighter(cropped_image):
                        poses_data.append(poses[i])
                else:
                    print('Empty box')
        return poses_data


def label_data(data, label):
    xy_data = data[:, :, :2]
    xy_data = xy_data.reshape((xy_data.shape[0], -1))
    column = np.ones((data.shape[0], 1)) * label
    labeled_data = np.hstack((xy_data, column))
    return labeled_data


# Main part of the script
yolo_model_path = "../yolov8m_custom.pt"
nas_pose_model_name = "yolo_nas_pose_l"

model_nas_pose = PoseDetector(nas_pose_model_name)
model_yolo = FighterDetector(yolo_model_path)

video_processor = VideoProcessor(model_nas_pose, model_yolo, take_every=10, limit=None)

train_false_path = "../video/train/false/kd-false - 1.mov"
train_true_path = "../video/train/true/kd-true - 1.mov"
val_false_path = "../video/val/false/kd-false - 2.mov"
val_true_path = "../video/val/true/vt-true - 1.mov"
test_false_path = "../video/test/Test false.mp4"
test_true_path = "../video/test/test_true.mov"

train_true_data = video_processor.process_video(train_true_path)
train_false_data = video_processor.process_video(train_false_path)
val_true_data = video_processor.process_video(val_true_path)
val_false_data = video_processor.process_video(val_false_path)
test_true_data = video_processor.process_video(test_true_path)
test_false_data = video_processor.process_video(test_false_path)

train_true_labeled = label_data(train_true_data, label=1)
train_false_labeled = label_data(train_false_data, label=0)
val_true_labeled = label_data(val_true_data, label=1)
val_false_labeled = label_data(val_false_data, label=0)
test_true_labeled = label_data(test_true_data, label=1)
test_false_labeled = label_data(test_false_data, label=0)

train_path = '../data/test.csv'
val_path = '../data/val.csv'
test_path = '../data/test.csv'

np.savetxt(train_path, np.vstack((train_true_labeled, train_false_labeled)), delimiter=';', fmt='%d')
np.savetxt(val_path, np.vstack((val_true_labeled, val_false_labeled)), delimiter=';', fmt='%d')
np.savetxt(test_path, np.vstack((test_true_labeled, test_false_labeled)), delimiter=';', fmt='%d')
