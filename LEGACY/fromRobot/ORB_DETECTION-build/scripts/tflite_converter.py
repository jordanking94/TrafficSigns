import tensorflow as tf

graph_file = 'retrained_graph.pb'
input_array = ["input"]
output_array = ["final_result"]

sess = tf.compat.v1.Session

converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_file, input_array, output_array)
converter.inference_type = tf.float32

#converter.inference_input_type = tf.float32#tf.uint8
#converter.inference_output_type = tf.float32
#converter.optimizations = [tf.compat.v1.lite.Optimize.OPTIMIZE_FOR_LATENCY]
#converter.optimizations = [tf.compat.v1.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
open("tf_files/graph.lite", "wb").write(tflite_model)
