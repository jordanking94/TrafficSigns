import numpy as np
import cv2
import tensorflow.compat.v1 as tf


input_arrays = ["input"]
output_arrays = ["final_result"]
graph_def_file = "./saved_model.pb"
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("model.lite", "wb").write(tflite_model)

