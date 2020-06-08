# Python
import cv2
import time
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS

# TF2 
import tensorflow as tf
from yolov3_tf2.models import (YoloV3)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs

# gRPC
import grpc
from concurrent import futures
from api import object_detection_pb2
from api import object_detection_pb2_grpc

ONE_DAY_IN_SECONDS = 60 * 60 * 24

# Model
WEIGHTS = "./checkpoints/yolov3.tf"
CLASSES = "./data/coco.names"

import cv2
import grpc
import time
import numpy as np
from concurrent import futures

from api import object_detection_pb2
from api import object_detection_pb2_grpc

ONE_DAY_IN_SECONDS = 60 * 60 * 24

class ObjectDetectionServicer(object_detection_pb2_grpc.ObjectDetectionServicer):
    def __init__(self):
        print("Initializied test service for tf2")
        # Get Yolov3 Model
        self.yolo = YoloV3(classes=80)
        self.yolo.load_weights(WEIGHTS)
        self.class_names = [c.strip() for c in open(CLASSES).readlines()]
        print("Loaded Yolov3 model and weights.")
    
    def objectDetection(self, request, data):
        # Receive img from gRPC client
        np_img = np.fromstring(request.data, np.uint8)
        cv_img = cv2.imdecode(np_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        print("Received the image with shape: ", cv_img.shape)
        
        cv_img = tf.expand_dims(cv_img, 0)
        cv_img = transform_images(cv_img, 416)
        
        # Get detection
        boxes, scores, classes, nums = self.yolo.predict(cv_img)
        
        objects=[]
        batch_idx = 0
        
        for i in range(nums[batch_idx]):
            label=self.class_names[int(classes[batch_idx][i])]
            probability=np.array(scores[batch_idx][i])
            xmin=int(np.array(boxes[batch_idx][i][0])*width)
            ymin=int(np.array(boxes[batch_idx][i][1])*height)
            xmax=int(np.array(boxes[batch_idx][i][2])*width)
            ymax=int(np.array(boxes[batch_idx][i][3])*height)

            new_obj = object_detection_pb2.Object(
                label=label,
                probability=probability,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )            
            objects.append(new_obj)
        return object_detection_pb2.Detection(objects=objects)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    object_detection_pb2_grpc.add_ObjectDetectionServicer_to_server(ObjectDetectionServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

def main(_argv):
    # Set GPU option (fixes some memory problems)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    # Run the service
    serve()

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass