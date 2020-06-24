import sys

# this is to ensure that everything will be able to work for MAC
#  note for future reference if google.protobuf issue: the __init__.py file is missing in site-packages/google directory. creating an empty __init__.py file there should work

sys.path.append("/usr/local/Cellar/python@2/2.7.16/Frameworks/Python.framework/Versions/2.7/lib/python27.zip")
sys.path.append("/usr/local/Cellar/python@2/2.7.16/Frameworks/Python.framework/Versions/2.7/lib/python2.7")
sys.path.append("/usr/local/Cellar/python@2/2.7.16/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-darwin")
sys.path.append("/usr/local/Cellar/python@2/2.7.16/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-mac")
sys.path.append("/usr/local/Cellar/python@2/2.7.16/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-mac/lib-scriptpackages")
sys.path.append("/usr/local/Cellar/python@2/2.7.16/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-tk")
sys.path.append("/usr/local/Cellar/python@2/2.7.16/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-old")
sys.path.append("/usr/local/Cellar/python@2/2.7.16/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-dynload")
sys.path.append("/Users/Jordan/Library/Python/2.7/lib/python/site-packages")
sys.path.append("/usr/local/lib/python2.7/site-packages")
sys.path.append("/usr/local/Cellar/numpy/1.16.3/libexec/nose/lib/python2.7/site-packages")


# -----------------------------------
# added
# -----------------------------------

import numpy as np
import tensorflow as tf
import time
import cv2

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

if __name__ == '__main__':
    tf.enable_eager_execution()
    model_path = "tf_files/graph.lite"
    
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_mean = 127.5
    input_std = 127.5

    labels = load_labels("tf_files/labels.txt")
