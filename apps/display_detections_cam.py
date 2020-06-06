import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs

WEIGHTS = "./checkpoints/yolov3.tf"
CLASSES = "./data/coco.names"

class DetectStream():
    def __init__(self):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)
        
        # Get Yolov3 Model
        self.yolo = YoloV3(classes=80)
        self.yolo.load_weights(WEIGHTS)
        self.class_names = [c.strip() for c in open(CLASSES).readlines()]
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.display_detections()
        
    
    def display_detections(self):
        start = time.time()
        tick = 0
        frame_counter = 0
        
        while True:
            _, cv_img = self.cap.read()
            if cv_img is None:
                logging.warning("Empty Frame")
                time.sleep(0.1)
                continue

            # img_tf = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            # img_tf = tf.expand_dims(img_tf, 0)
            # img_tf = transform_images(img_tf, 416)

            # boxes, scores, classes, nums = self.yolo.predict(img_tf)
            # cv_img = draw_outputs(cv_img, (boxes, scores, classes, nums), self.class_names)
            
            
            # Display the detection
            cv2.imshow('output', cv_img)
            if cv2.waitKey(1) == ord('q'):
                break
            
            # Print FPS
            frame_counter += 1
            time_now = time.time() - start
            if(time_now - tick >= 1):
                tick += 1
                print(cv_img.shape)
                print("FPS:", frame_counter)
                frame_counter = 0

def main(_argv):
    DetectStream()

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass








# def main(_argv):
 
#     physical_devices = tf.config.experimental.list_physical_devices('GPU')
#     for physical_device in physical_devices:
#         tf.config.experimental.set_memory_growth(physical_device, True)
    
#     yolo = YoloV3(classes=80)
#     # self.yolo = YoloV3()
#     yolo.load_weights(WEIGHTS)
#     class_names = [c.strip() for c in open(CLASSES).readlines()]
#     cap = cv2.VideoCapture(0)

#     while True:
#         _, img = cap.read()

#         if img is None:
#             logging.warning("Empty Frame")
#             time.sleep(0.1)
#             continue
        
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = tf.expand_dims(img, 0)
#         img = transform_images(img, 416)

#         boxes, scores, classes, nums = yolo.predict(img)
#         print(nums) 
#         img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        
#         # Display the detection
#         cv2.imshow('output', img)
#         if cv2.waitKey(1) == ord('q'):
#             break
    
# if __name__ == '__main__':
#     try:
#         app.run(main)
#     except SystemExit:
#         pass

