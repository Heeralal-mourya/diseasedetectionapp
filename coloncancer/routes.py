from flask import Flask, render_template, url_for, redirect, request, jsonify, Response 
from coloncancer import app
from coloncancer import classifier,validation_classifier
from datetime import datetime
from coloncancer.util import save_image, get_preprocessed_img
import numpy as np
from PIL import Image
import os
import logging
import cv2
#import requests
#import pandas as pd 
import time
import matplotlib.pyplot as plt

height = 150
width = 150
dim = 3
val_obj = validation_classifier.val_model()
model_obj = classifier.coloncancer_Model()

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('colon-cancer.html')
    
@app.route("/coloncancerdetect", methods=['GET', 'POST'])
def coloncancerdetect():
    try:
        temp_id = request.form.get("temp_id")
        timestamp = str(time.time()).split(".")[0]
        img_request = request.files.get('image_file')
    except Exception as msg:
        print("[ERROR] in reading request form and files : " + time.asctime(time.localtime(time.time())) + " : " + str(msg))
        data = {'temp_id':temp_id,'status' : False, 'message' : str(msg)}
        resp = jsonify(data)
        resp.status_code = 200
        return resp
    try:
        img_name = temp_id + "_" + timestamp + "_" + img_request.filename
        img_file_name = save_image(img_request, img_name)
    except Exception as msg:
        print("[ERROR] while saving image: " + time.asctime(time.localtime(time.time())) + " : " + str(msg))
        data = {'temp_id':temp_id,'status' : False, 'message' : str(msg)}
        resp = jsonify(data)
        resp.status_code = 200
        return resp 
    try:
        img_path = os.path.join(app.root_path, 'static/Uploaded', img_file_name)
        val_result = val_obj.predict(img_path)
        print(val_result)
        if (val_result=='negative'):                
            raise Exception('Invalid Image')
        img = get_preprocessed_img(img_path, height, width, dim)   
    except Exception as msg:
        print("[ERROR] while preprocessing image : " + time.asctime(time.localtime(time.time())) + " : " + str(msg))
        data = {'temp_id':temp_id,'status' : False, 'message' : str(msg)}
        resp = jsonify(data)
        resp.status_code = 200
        return resp 
    try:
        result = model_obj.predict(img)
    except Exception as msg:
        print("[ERROR] while prediction : " + time.asctime(time.localtime(time.time())) + " : " + str(msg))
        data = {'temp_id':temp_id,'status' : False, 'message' : str(msg)}
        resp = jsonify(data)
        resp.status_code = 200
        return resp 
    
    resp_dict = {'temp_id':temp_id, 'status':True, 'Adipose':'{:.2f}'.format(result[0][0]),'Complex':'{:.2f}'.format(result[0][1]), 'Debris':'{:.2f}'.format(result[0][2]),     'Empty':'{:.2f}'.format(result[0][3]),'Lympho':'{:.2f}'.format(result[0][4]),'Mucosa':'{:.2f}'.format(result[0][5]),'Storma':'{:.2f}'.format(result[0][6]), 'Tumor':'{:.2f}'.format(result[0][7])}        
    resp = jsonify(resp_dict)
    resp.status_code = 200
    return resp
    
