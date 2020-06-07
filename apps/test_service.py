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
        print("Initializied test service")
    
    def objectDetection(self, request, data):
        # Receive img from gRPC client
        np_img = np.fromstring(request.data, np.uint8)
        cv_img = cv2.imdecode(np_img, cv2.COLOR_BGR2RGB)
        print("Received the image with shape: ", cv_img.shape)
        
        objects = []

        new_obj = object_detection_pb2.Object(
            label = "test",
            probability = 100,
            xmin = 100,
            ymin = 100,
            xmax = 200,
            ymax = 200,
        )
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

if __name__ == '__main__':
    serve()