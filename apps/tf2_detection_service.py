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
    
    def objectDetection(self, request, data):
        # Receive img from gRPC client
        np_img = np.fromstring(request.data, np.uint8)
        cv_img = cv2.imdecode(np_img, cv2.COLOR_BGR2RGB)
        print("Received the image with shape: ", cv_img.shape)
        
        objects = []
        
        # Create a test BBox
        new_obj = object_detection_pb2.Object(
            label = "test_tf2",
            probability = 100,
            xmin = 100,
            ymin = 100,
            xmax = 200,
            ymax = 200,
        )
        # Return the test detection
        objects.append(new_obj)
        return object_detection_pb2.Detection(objects=objects)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    object_detection_pb2_grpc.add_ObjectDetectionServicer_to_server(ObjectDetectionServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

def main(_argv):
    serve()

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass