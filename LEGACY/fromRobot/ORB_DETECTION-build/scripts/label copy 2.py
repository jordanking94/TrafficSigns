# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

if __name__ == "__main__":
    start = time.time()
    file_path = "tmp/1.jpg"
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = cv2.imread(file_path)
    resized = cv2.resize(img, (width, height))
    
    floating_model = False
    
    if input_details[0]['dtype'] == type(np.float32(1.0)):
        floating_model = True

    # add N dim
    input_data = np.expand_dims(resized, axis=0)

    input_data = np.float32(input_data)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
        
    interpreter.set_tensor(input_details[0]['index'], input_data)
        
    interpreter.invoke()
        
        
    output_data = interpreter.get_tensor(output_details[0]['index'])

        
    output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # removes signle-dimension entries from shape of array
        # the shape of the input tensor is [1 128 128 3]
    results = np.squeeze(output_data)
        
#print labels
#print results
        
    end=time.time()
#print('\nLABEL: {:.5f} seconds'.format(end-start))

    f= open("tmp/2.txt","w+")

    max_val = max(results)
    max_ind = np.where(results == max_val)

#print max_ind[0]
#print results[max_ind]

    f.write("%d\n%f\n" % (max_ind[0],results[max_ind]))
    f.close()






