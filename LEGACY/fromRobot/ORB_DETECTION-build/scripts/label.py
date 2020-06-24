

import numpy as np
import tensorflow as tf
#import cv2
#print("here1")
#print("here1")
#print("here1")

#tf.enable_eager_execution()
#tf.enable_eager_execution()
#tf.compat.v1.enable_eager_execution
tf.enable_eager_execution()
model_path = "tf_files/graph.lite"
print(tf.__version__)
    
# Load TFLite model and allocate tensors.
#interpreter = tf.lite.Interpreter(model_path=model_path)

#interpreter = tf.lite.Interpreter(model_path=path)
interpreter = tf.contrib.lite.Interpreter(model_path=model_path)
print("here")
interpreter.allocate_tensors()

    
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input = interpreter.tensor(input_details[0]["index"])
output = interpreter.tensor(output_details[0]["index"])

mean = 128
std_dev = 127

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = False
if input_details[0]['dtype'] == type(np.float32(1.0)):
    floating_model = True


def label(img_data,img_height, img_width):
    #print(img_height)
    #print(img_width)
    image = np.asarray(img_data,dtype="uint8")
    image = np.reshape(image, (1,img_height, img_width, 3))
    
    #cv2.imwrite("image.jpg", image)
    #cv2.waitKey(0)

    if floating_model:
        #print("floating model")
        image = np.float32(image)
        image = (image - mean) / std_dev
    
    
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()


#print("hello-1")
    #output_tensor = interpreter.get_tensor(output_details[0]['index'])

#print(output_tensor)

#predictions = np.squeeze(output_tensor)
    predictions = np.squeeze(output()[0])
#print(predictions)
    predicted_confidence = max(predictions)
#print("hello0")
    object_class = np.where(predictions==predicted_confidence)

#print("hello1")
    object_class = object_class[0][0]
#print("hello2")

#print(object_class)
#print(predicted_confidence)
    return (object_class, predicted_confidence)
