# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import base64
import cv2
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import random



def set_background(image_file):
    
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
            filter:blur(8px)
            -webkit-filter:blur(8px)
            
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

    


def classify(image, model, class_names,H):
    
    # convert image to (224, 224)
    image = ImageOps.fit(image,(H,H),Image.Resampling.LANCZOS)
    image_array = np.array(image)
    
    data = np.ndarray(shape=(1,H,H,3),dtype=np.float32)
    data[0] = image_array
    # make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

    


