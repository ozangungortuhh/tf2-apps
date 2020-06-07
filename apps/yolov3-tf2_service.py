import time
import cv2
import numpy as np
import grpc
from concurrent import futures
from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf

from yolov3_tf2.models import (YoloV3)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs


import api.darknet_detection_pb2 as darknet_detection_pb2
import api.darknet_detection_pb2_grpc as darknet_detection_pb2_grpc

WEIGHTS = "./checkpoints/yolov3.tf"
CLASSES = "./data/coco.names"

class DarknetDetectionServicer(darknet_detection_pb2_grpc.DarknetDetectionServicer):
    
    def __init__(self):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)
        
        # Get Yolov3 Model
        self.yolo = YoloV3(classes=80)
        self.yolo.load_weights(WEIGHTS)
        self.class_names = [c.strip() for c in open(CLASSES).readlines()]
        print("Yolov3-tf2 service is initialized.")

    def darknetDetection(self, request, context):
        # Receive img from gRPC client
        np_img = np.fromstring(request.data, np.uint8)
        cv_img = cv2.imdecode(np_img, cv2.COLOR_BGR2RGB)
        
        height, width, channels = cv_img.shape
        print(f"Received image with height: {height} and width {width}") 
        
        # Convert img to tensorflow format
        img_tf = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_tf = tf.expand_dims(img_tf, 0)
        img_tf = transform_images(img_tf, 416)
        
        # Apply detection
        boxes, scores, classes, nums = self.yolo.predict(img_tf)
        objects = []
        batch_idx = 0
        
        for i in range(nums[batch_idx]):
            label=self.class_names[int(classes[batch_idx][i])]
            probability=np.array(scores[batch_idx][i])
            xmin=int(np.array(boxes[batch_idx][i][0])*width)
            ymin=int(np.array(boxes[batch_idx][i][1])*height)
            xmax=int(np.array(boxes[batch_idx][i][2])*width)
            ymax=int(np.array(boxes[batch_idx][i][3])*height)

            new_obj = darknet_detection_pb2.Object(
                label=label,
                probability=probability,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )            
            objects.append(new_obj)
        
        print(objects)
        return darknet_detection_pb2.Detection(objects=objects)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    darknet_detection_pb2_grpc.add_DarknetDetectionServicer_to_server(DarknetDetectionServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            continue
    except KeyboardInterrupt:
        server.stop(0)

def main(_argv):
    serve()

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
