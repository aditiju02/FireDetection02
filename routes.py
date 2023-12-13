from flask import json, jsonify, render_template, request, redirect, url_for, session, flash
import os
from werkzeug.utils import secure_filename
# from application.db import Images
from application import app

import cv2
import numpy as np
import base64
import smtplib
import ssl
from email.message import EmailMessage

# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
#matplotlib inline  
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#preprocess.
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from keras.models import load_model

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
# from tqdm import tqdm
import os                   
# from random import shuffle  
# from zipfile import ZipFile
from PIL import Image

# from keras.applications import VGG16
IMG_SIZE=256

@app.route('/')
def hello():
    return render_template('index.html')

def process_image(image_data):
    # Load the image from base64-encoded data
    image_bytes = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # call model prediction function
    # model = load_model('C:\Users\majge\Downloads\FireDetection\FireDetection\application\static\model.h5')
    model = load_model('application\static\model1.h5')
    
    x = Image.fromarray(img)
    x=x.resize((IMG_SIZE,IMG_SIZE))
    x=x.convert('RGB')
    plt.imshow(x)
    x = img_to_array(x)
    x= segmentation(x)
    # plt.imshow(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    print(array)
    result = array[0]
    answer = np.argmax(result)
    print("Answer: ")
    print(answer)  

    if str(answer) == "1":
        data = "1"
        return data
    
    elif str(answer) == "0": 
        data = "0"
        return data
    
def segmentation(arr):
    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            r=arr[i,j,0]
            g=arr[i,j,1]
            b=arr[i,j,2]
            if(r>g and g>b and r>200):
                pass
            else:
                arr[i,j]=[0,0,0]
    return arr

@app.route('/scan', methods=['POST'])
def scan():
    try:
        image_data = request.json.get('image_data')
        data = process_image(image_data)
        print(data)
        if data:
            mailsent = sendmail("basvashri.majge@mitaoe.ac.in")
            print(mailsent)
            return jsonify({'data': data})
        else:
            return jsonify({'error': 'Not working'})
    except Exception as e:
        return jsonify({'error': str(e)})
    
def sendmail(email):
    
    # Enter the email id of the sender
    sender = 'majgebasvashri@gmail.com'

    # security key
    password = 'hduf tyyo lymc brrq'

    #Enter subject of email
    subject = "💀💀🚨🚨From Fire Detection System"

    #Enter the body of email 
    body = "Fire Detected!!!💀💀🚨🚨 Fire Detected!!!💀💀🚨🚨 Fire is detected at your house, please hurry"

    receiver = email

    em = EmailMessage()
    em['From'] = sender
    em['To'] = receiver
    em['Subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context = context) as smtp:
        smtp.login(sender, password)
        smtp.sendmail(sender, receiver, em.as_string())
    
    return "sent"