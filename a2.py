# import the necessary packages
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2
import tensorflow as tf
model = tf.keras.models.load_model('C:\Users\Administrateur\Desktop\pfe\tifinagh.reco\model_3.h5')


list ={0:'ya',1:'yab',2:'yac',3:'yad',4:'yadd',5:'yae',6:'yaf',7:'yag',8:'yagh',9:'yagw',10:'yah',11:'yahh',12:'yaj',13:'yak',14:'yakw',15:'yal',16:'yam',17:'yan',18:'yaq',19:'yar',20:'yarr',21:'yas',22:'yass',23:'yat',24:'yatt',25:'yaw',26:'yax',27:'yay',28:'yaz',29:'yazz',30:'yey',31:'yi',32:'yu'}

list2 = ["ⴰ","ⴱ","ⵛ","ⴷ","ⴹ","ⵄ","ⴼ","ⴳ","ⵖ","ⴳⵯ","ⵀ","ⵃ","ⵊ","ⴽ","ⴽⵯ","ⵍ","ⵎ","ⵏ","ⵇ","ⵔ","ⵕ","ⵙ","ⵚ","ⵜ","ⵟ","ⵡ","ⵅ","ⵢ","ⵣ","ⵥ","ⴻ","ⵉ","ⵓ"]

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#matplotlib inline

#N = "%02d" % i
print (PATH)
image = mpimg.imread(PATH) # images are color images
#plt.show()
imgsize = 128
img = cv2.imread(PATH,cv2.IMREAD_COLOR)
img =image[:,:]

img = cv2.resize(img,(imgsize,imgsize))
plt.imshow(img)
plt.show()
img = img.reshape(1,128,128,3)
img.shape

#{'x': [98, 268, 391, 411], 'y': [64, 66, 93, 67], 'w': [113, 91, 7, 85], 'h': [146, 133, 8, 117]}


#plt.imshow(image)

a = model.predict(img)
y_pred = a.argmax()
print("This image most likely belongs to {} / {} with a {:.2f} % percent confidence.".format(list[y_pred],list2[y_pred], 100 * np.max(a)))



#list[y_pred]
#list2[y_pred]


#prediction = model.predict(img)
#prediction_probability = np.amax(a)
#prediction_idx = np.argmax(prediction)
#list[prediction_idx-1]
#prediction_idx = np.argmax(prediction)
#list[prediction_idx]
#np.argmax(a)
