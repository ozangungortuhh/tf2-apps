import cv2
import numpy as np
import tensorflow as tf
from concurrent import futures
import os
from absl import app, flags, logging


class DetectStream():
    def __init__(self):
      # Get model
      self.model = tf.keras.models.load_model('models/efficientnet_udt')
      # Get stream
      self.cap = cv2.VideoCapture(0)
      self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
      self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      # Get detection
      detect_image()

    def detect_image(self, request, context):
      while True:
        _, cv_img = self.cap.read()
        if cv_img is None:
          logging.warning("Empty Frame")
          time.sleep(0.1)
          continue

        cv_img = np.expand_dims(cv_img, axis = 0)

        result = self.model.predict(cv_img)
        print(results)
        # predicted_class = np.argmax(result[0], axis=-1)

if __name__ == '__main__':
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

    DetectStream()