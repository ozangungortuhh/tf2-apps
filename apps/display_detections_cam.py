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


flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

WEIGHTS = "./checkpoints/yolov3.tf"
CLASSES = "./data/coco.names"

def main(_argv):
 
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)
    
    yolo = YoloV3(classes=80)
    # self.yolo = YoloV3()
    yolo.load_weights(WEIGHTS)
    class_names = [c.strip() for c in open(CLASSES).readlines()]
    cap = cv2.VideoCapture(0)

    while True:
        _, img = cap.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue
        
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        boxes, scores, classes, nums = yolo.predict(img_in)
        print(nums) 
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        
        # Display the detection
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

