import streamlit as st
import imutils
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
model = tf.keras.models.load_model('C:/Users/Administrateur/Desktop/pfe/tifinagh.reco/model_3.h5')


list ={0:'ya',1:'yab',2:'yac',3:'yad',4:'yadd',5:'yae',6:'yaf',7:'yag',8:'yagh',9:'yagw',10:'yah',11:'yahh',12:'yaj',13:'yak',14:'yakw',15:'yal',16:'yam',17:'yan',18:'yaq',19:'yar',20:'yarr',21:'yas',22:'yass',23:'yat',24:'yatt',25:'yaw',26:'yax',27:'yay',28:'yaz',29:'yazz',30:'yey',31:'yi',32:'yu'}

list2 = ["ⴰ","ⴱ","ⵛ","ⴷ","ⴹ","ⵄ","ⴼ","ⴳ","ⵖ","ⴳⵯ","ⵀ","ⵃ","ⵊ","ⴽ","ⴽⵯ","ⵍ","ⵎ","ⵏ","ⵇ","ⵔ","ⵕ","ⵙ","ⵚ","ⵜ","ⵟ","ⵡ","ⵅ","ⵢ","ⵣ","ⵥ","ⴻ","ⵉ","ⵓ"]

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import pandas as pd
import urllib

# set page layout
st.set_page_config(
    page_title="Recognition of handwritten tifinagh characters",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Image Classification")
st.sidebar.subheader("Input")
models_list = ["Word","Character"]
network = st.sidebar.selectbox("Select the Model", models_list)
st.write(network)
model = tf.keras.models.load_model('C:/Users/Administrateur/Desktop/pfe/tifinagh.reco/model_3.h5')

# define a dictionary that maps model names to their classes
# inside Keras
MODELS = {
    "Word": model,
    "Character": model,
}
uploaded_file = st.sidebar.file_uploader("Choose an image to classify", type=["jpg", "jpeg"])
if network == "Word":
    
    # component to upload images
    # component for toggling code
    #show_code = st.sidebar.checkbox("Show Code")


    def sort_contours(cnts, method="left-to-right"):
        reverse = False
        i = 0
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)
        
    def zone (d):
      ts={'X':[],'Y':[]}

      if len(d['x'])==1 :
            ts['X'].append(0)
            ts['X'].append(d['x'][0]+d['w'][0]+50)
            ts['Y'].append(0)
            ts['Y'].append(d['y'][0]+d['h'][0]+50 )
      else :
            for i in range (0,len(d['x'])):
                if i==0 :
                  ts['X'].append(0)
                  ts['X'].append(d['x'][1]-10)

                else :
                    if i==len(d['x'])-1:
                      ts['X'].append(d['x'][i-1]+d['w'][i-1]+10)
                      ts['X'].append(d['x'][i]+d['w'][i]+25)
                    else :  
                      ts['X'].append(d['x'][i-1]+d['w'][i-1]+10)      
                      ts['X'].append(d['x'][i+1]-10)

          #ts['X'].append(max(d['x'])+20)
            ts['Y'].append(min(d['y'])-20)
            ts['Y'].append(max(d['h'])+20)
      return ts
      
    def seg(d) :
      a=0
      dim =[]
      for i in range (0,len(zone(d)['X']),2):
        dim.append((zone(d)['X'][i],zone(d)['X'][i+1]))
        a=a+1
      return a , dim
      
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    #%matplotlib inline

    def segmentation(PATH , d) :
      a , dim = seg(d)

      #print (PATH)
      image = mpimg.imread(PATH) # images are color images
      imgsize = 128
      img = cv2.imread(PATH,cv2.IMREAD_COLOR)
      p=[]
      for i in range (0,a):
        img =image[d['y'][i]-30:d['y'][i]+d['h'][i]+30, dim[i][0]:dim[i][1]]
        img = cv2.resize(img,(imgsize,imgsize))
        #plt.imshow(img)

        #plt.show()
        img = img.reshape(1,128,128,3)
        #img.shape
        pred = model.predict(img,verbose=0)
        y_pred = pred.argmax()
        #print("This image most likely belongs to {} / {} with a {:.2f} % percent confidence.".format(list[y_pred],list2[y_pred], 100 * np.max(pred)))
        p.append(list2[y_pred])
      st.write("This image most likely belongs to :  "+''.join(p))
      
    def get_letters(img):
        letters = []
        image = cv2.imread(img)
        threshs = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
        y =0 
        h=0
        x=0
        w=0
        ret,thresh1 = cv2.threshold(gray ,  100,255,cv2.THRESH_BINARY_INV)
        dilated = cv2.dilate(thresh1, None, iterations=2)

        cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sort_contours(cnts, method="left-to-right")[0]
            # loop over the contours
        d = {'x':[],'y':[],'w':[],'h':[]}
        for c in cnts:
            if cv2.contourArea(c) > 10:
                (x, y, w, h) = cv2.boundingRect(c)
                if h<50 :
                  continue
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                d['x'].append(x)
                d['y'].append(y)
                d['h'].append(h)
                d['w'].append(w)
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = cv2.resize(thresh, (128, 128), interpolation = cv2.INTER_CUBIC)
            thresh = thresh.astype("float32") / 255.0
            thresh = np.expand_dims(thresh, axis=-1)
            thresh = cv2.resize(thresh,(128,128))
            thresh = thresh.reshape(1,128,128)
            thresh = thresh.reshape(1,128,128,1)
            thresh = np.concatenate((thresh,thresh,thresh), axis=-1)
            threshs.append(thresh)
            list ={0:'ya',1:'yab',2:'yac',3:'yad',4:'yadd',5:'yae',6:'yaf',7:'yag',8:'yagh',9:'yagw',10:'yah',11:'yahh',12:'yaj',13:'yak',14:'yakw',15:'yal',16:'yam',17:'yan',18:'yaq',19:'yar',20:'yarr',21:'yas',22:'yass',23:'yat',24:'yatt',25:'yaw',26:'yax',27:'yay',28:'yaz',29:'yazz',30:'yey',31:'yi',32:'yu'}
            
      

        return letters, image , thresh  , threshs, d

    #plt.imshow(image)


    def get_word(letter):
        word = "".join(letter)
        return word
        



    if uploaded_file:
        bytes_data = uploaded_file.read()
        st.write(uploaded_file.name)


        model = tf.keras.models.load_model('C:/Users/Administrateur/Desktop/pfe/tifinagh.reco/model_3.h5')

        PATH = "C:/Users/Administrateur/Desktop/test/"+uploaded_file.name
        #N = "%02d" % i
        print (PATH)
        letter,image , thresh , threshs , d = get_letters(PATH)
        word = get_word(letter)
        st.write(word)
        st.image(image,width=600,caption='Image to be predicted')
        #image = mpimg.imread(PATH) # images are color images
        #plt.show()
        #imgsize = 128
        #img = cv2.imread(PATH,cv2.IMREAD_COLOR)
        #img =image[:,:]

        #img = cv2.resize(img,(imgsize,imgsize))
        #plt.imshow(img)
        #plt.show()
        #img = img.reshape(1,128,128,3)
        #img.shape
        segmentation(PATH,d)

        #preds = model.predict(img)
        
        #y_pred = preds.argmax()
        #st.write("This image most likely belongs to {} / {} with a {:.2f} % percent confidence.".format(list[y_pred],list2[y_pred], 100 * np.max(preds)))
        #st.image(img,width=200,caption='Image to be predicted')


        









    #import matplotlib.pyplot as plt
    #import matplotlib.image as mpimg

    #matplotlib inline

    #PATH = "C:/Users/adilo/Downloads/2.jpg"
    #N = "%02d" % i
    #print (PATH)
    #image = mpimg.imread(PATH) # images are color images
    #plt.show()
    #imgsize = 128
    #img = cv2.imread(PATH,cv2.IMREAD_COLOR)
    #img =image[:,:]

    #img = cv2.resize(img,(imgsize,imgsize))
    #plt.imshow(img)
    #plt.show()
    #img = img.reshape(1,128,128,3)
    #img.shape

    #{'x': [98, 268, 391, 411], 'y': [64, 66, 93, 67], 'w': [113, 91, 7, 85], 'h': [146, 133, 8, 117]}


    #plt.imshow(image)

    #a = model.predict(img)
    #y_pred = a.argmax()
    #print("This image most likely belongs to {} / {} with a {:.2f} % percent confidence.".format(list[y_pred],list2[y_pred], 100 * np.max(a)))

else :
    # component to upload images
    # component for toggling code
    #show_code = st.sidebar.checkbox("Show Code")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    #matplotlib inline
    if uploaded_file:
        bytes_data = uploaded_file.read()
        st.write(uploaded_file.name)
        PATH = "C:/Users/Administrateur/Desktop/test/"+uploaded_file.name
        model = tf.keras.models.load_model('C:/Users/Administrateur/Desktop/pfe/tifinagh.reco/model_3.h5')
    #PATH = "/content/7.jpg"
    #N = "%02d" % i
    #print (PATH)
        image = mpimg.imread(PATH) # images are color images
        st.write(image.shape)
        if st.button('rotate'):
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


    #plt.show()
        img =image[ :, :]
        imgsize = 128
        img = cv2.resize(img,(imgsize,imgsize))

        #plt.show()
        img = img.reshape(1,128,128,3)
        st.image(img)
        img.shape
        pred = model.predict(img,verbose=0)
        y_pred = pred.argmax()

        
        #imgsize = 128
        #img = cv2.imread(PATH,cv2.IMREAD_COLOR)
        #img =image[ :, :]
        #img = cv2.resize(img,(imgsize,imgsize))
        #plt.imshow(img)
        #plt.show()
        #img = img.reshape(1,128,128,3)
        #img.shape
        pred = model.predict(img,verbose=0)
        y_pred = pred.argmax()
        st.write("This image most likely belongs to {} / {} with a {:.2f} % percent confidence.".format(list[y_pred],list2[y_pred], 100 * np.max(pred)))


