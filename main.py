# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 20:46:19 2023

@author: Pranav
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from PIL import Image,ImageOps
import numpy as np
import keras.backend as K
import cv2
from util import set_background
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def dice_coeff(y_true, y_pred):
    smooth = 1e-15
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return ((2. * intersection+smooth) / (K.sum(y_true) + K.sum(y_pred)+smooth))

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coeff(y_true, y_pred)

def tversky(y_true, y_pred):
    smooth_tversky=1e-6
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth_tversky)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth_tversky)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)
    #pt_1 = tversky(y_true, y_pred)
    #print(pt_1)
    gamma = 0.75
    return K.pow((1-tversky(y_true, y_pred)), gamma)

set_background('C:/Users/Pranav/Downloads/BG_wall(1).png')

# set title
original_title = '<center><p style="border-radius:10px;background-color:#C0C0C099;font-family:bebas neue; color:black; font-size: 50px;">AI MEDICAL DIAGNOTICS ASSISTANT</p></center>'
st.markdown(original_title, unsafe_allow_html=True)

# set header
subt = '<center><p style="border-radius:10px;background-color:#C0C0C099;font-family:bebas neue; color:black; font-size: 40px;">Please upload a brain tumor OR a Chest CT scan image here (Other parts coming soon)</p></center>'
st.markdown(subt, unsafe_allow_html=True)
# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

brain = st.button("BRAIN TUMOR DETECTION")
brain_SEG = st.button("BRAIN TUMOR SEGMENTATION")
chest = st.button("CHEST CT COVID DETECTION")
chest_cancer = st.button("CHEST CT CANCER DETECTION")
Kidney = st.button("Kidney Diagnosis")
model_1 = load_model('C:/Users/Pranav/Downloads/Brain_tumour_model (1).h5',compile= False)
# load classifier
H = 128

if brain == True and chest == False and brain_SEG == False and chest_cancer == False and Kidney == False:
    model_1 = load_model('C:/Users/Pranav/Downloads/Brain_tumour_model (1).h5',compile= False)
    Class_list = ['Glioma Tumor', 'Meningioma Tumor', 'No tumor', 'Pituitary Tumor']
    model_1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'], sample_weight_mode='temporal')
    
if brain == False and chest == True and brain_SEG == False and chest_cancer == False and Kidney == False :
    model_1 = load_model('C:/Users/Pranav/Downloads/CHEST_CT_MODEL.h5',compile= False)
    Class_list = ['Covid','Normal']
    model_1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'], sample_weight_mode='temporal')

if brain == False and chest == False and brain_SEG == True and chest_cancer == False and Kidney == False:
    model_1 = load_model('C:/Users/Pranav/Downloads/brain_tumor_segmentation_model_1.h5',compile= False)
    Class_list = []
    model_1.compile(optimizer='adam',loss=focal_tversky,metrics=[dice_loss,'accuracy'], sample_weight_mode='temporal')
    
if brain == False and chest == False and brain_SEG == False and chest_cancer == True and Kidney == False:
    model_1 = load_model('C:/Users/Pranav/Downloads/CHEST_CANCER_MODEL.h5',compile= False)
    Class_list = ['Adenocarcinoma','Large cell Carcinoma','Normal','Squamous cell Carcinoma']
    model_1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'], sample_weight_mode='temporal')
    H = 256
    
if brain == False and chest == False and brain_SEG == False and chest_cancer == False and Kidney == True:
    model_1 = load_model('C:/Users/Pranav/Downloads/KIDNEY_DIAGNOSIS_MODEL.h5',compile= False)
    Class_list = ['Cyst','Normal','Stone','Tumor']
    model_1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'], sample_weight_mode='temporal')
    

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

    
# display image
if brain_SEG!=True:
    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)
        # classify image
        class_name, conf_score = classify(image, model_1, Class_list,H)
    
        # write classification
        st.write("## {}".format(class_name))
        st.write("### score: {}%".format(int(conf_score * 1000) / 10))
    else:
        st.error('PLEASE UPLOAD AN IMAGE OF THE FORMAT JPG,JPEG OR PNG', icon="🚨")
        
        
def segment(image,model):
    
    image = ImageOps.fit(image,(256,256),Image.Resampling.LANCZOS)
    img = np.array(image)
    img = img/255.0
    data = np.ndarray(shape=(1,256,256,3),dtype=np.float32)
    data[0] = img
    
    y_pred = model.predict(data)
    y_pred = np.squeeze(y_pred,axis = 0)
    y_pred = y_pred>=0.5
    y_pred = y_pred * 255.0
    y_pred = y_pred.astype(np.float32)
    return y_pred

if brain_SEG == True and chest_cancer!=True:
    
    if file is not None:
        
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)
        mask = segment(image,model_1)
        cv2.imwrite("C:/Users/Pranav/Downloads/temp_brain_mask.jpg",mask)
        mask_img = Image.open('C:/Users/Pranav/Downloads/temp_brain_mask.jpg').convert('RGB')
        image = ImageOps.fit(image,(256,256),Image.Resampling.LANCZOS)
        image = image.convert("RGBA")
        mask_img = mask_img.convert("RGBA")
        image = image.convert("RGBA")
        mask_img = mask_img.convert("RGBA")
        
        overlay_img = Image.blend(image, mask_img, 0.75)
        overlay_img.save("temp_brain_overlay.png","PNG")
        st.image('temp_brain_overlay.png',use_column_width=True)
        
    else:
        st.error('PLEASE UPLOAD AN IMAGE OF THE FORMAT JPG,JPEG OR PNG', icon="🚨")
    
    

    