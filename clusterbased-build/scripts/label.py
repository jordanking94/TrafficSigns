

import numpy as np
import tensorflow as tf
import cv2 as cv

tf.compat.v1.enable_eager_execution
model_path = "tf_files/graph.lite"

interpreter = tf.compat.v2.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

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


def label(img_data, x, y, xs, ys):
    image = np.asarray(img_data,dtype="uint8")
    image = np.reshape(image, (240, 320, 3))
    #print('x: ' + str(x) + ', y: ' + str(y) + 'xs: ' + str(xs) +', ys: ' + str(ys))
    sub_image = image[y:y+ys, x:x+xs,:]
    sub_image = cv.cvtColor(sub_image, cv.COLOR_BGR2RGB)
    sub_image = cv.resize(sub_image, (128,128))
    sub_image = np.reshape(sub_image, (1,128, 128, 3))
    #print(sub_image.shape)
    #image = shaped_full_image[1,x:xs,y:ys,:]
    #cv.imshow('display', sub_image )
    #cv.waitKey(0)
    
    
    sub_image = np.reshape(sub_image, (1,128, 128, 3))
    #image = full_image[0:128, 0:128]
    
    #image = np.reshape(image, (1,img_height, img_width, 3))

    if floating_model:
        sub_image = np.float32(sub_image)
        sub_image = (sub_image - mean) / std_dev
    interpreter.set_tensor(input_details[0]['index'], sub_image)

    interpreter.invoke()
    predictions = np.squeeze(output()[0])
    predicted_confidence = max(predictions)
    object_class = np.where(predictions==predicted_confidence)
    object_class = object_class[0][0]
    return (object_class, predicted_confidence)
