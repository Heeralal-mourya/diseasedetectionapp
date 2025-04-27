import os
import tensorflow as tf
import numpy as np
import cv2
from flask import current_app
import cv2
from skimage import data, color
from skimage.transform import rescale, resize
from skimage.io import imread 
from tensorflow.keras.preprocessing import image   

#saving image in static/Uploaded
def save_image(cur_img, picture_name):

    cur_img.save(os.path.join(current_app.root_path, 'static/Uploaded', picture_name))
    return picture_name

#To load image from static/Uploaded with required dimensions
def get_preprocessed_img(image_path, height, width, dim):
	# loads RGB image as PIL.Image.Image type
	img = image.load_img(image_path, target_size=(height,width))
	# convert PIL.Image.Image type to 3D tensor with shape (150, 150, 3)
	img = image.img_to_array(img)
	img = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
	return img
    
