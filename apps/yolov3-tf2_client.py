import cv2
import numpy as np
import grpc
from concurrent import futures
import sys,os

import api.darknet_detection_pb2 as darknet_detection_pb2
import api.darknet_detection_pb2_grpc as darknet_detection_pb2_grpc

class Yolov3TF2Client():
    def __init__(self):
        self._yolov3tf2_channel = grpc.insecure_channel('localhost:50051')
        self._yolov3tf2_stub = darknet_detection_pb2_grpc.DarknetDetectionStub(self._yolov3tf2_channel)
        print("initialized yolov3-tf2 channel and stub")
    
    def test(self):
        cv_img = cv2.imread("/assets/dog.jpg")
        _ , img_jpg = cv2.imencode('.jpg', cv_img)
        image_msg = darknet_detection_pb2.Image(data=img_jpg.tostring())
        self._yolov3tf2_stub.darknetDetection(image_msg)
        print("Sent image to the service")

if __name__ == "__main__":
    yolov3tf2 = Yolov3TF2Client()
    yolov3tf2.test()