#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:47:11 2024

@author: kudu
"""

from utilities.model_utils import ModelUtils

from constants import const

import numpy as np
import cv2





class LivenessRecognitionModelUtils(ModelUtils):
    
    def __init__(self, model_path):
        super().__init__(model_path)
        
    
    
    
    
    def predict_eye_status(self, eye):
        img = cv2.resize(eye, (const.EYEIMG_SIZE, const.EYEIMG_SIZE))
        img = img.astype('float32')
        img /= 255
        
        # Add the batch dimension
        img = np.expand_dims(img, axis=0)
        
        prediction = self.model.predict(img)
        if prediction[0] > 0:
            return 1
        return 0

    
    
    
    def predict_left_eye_status(self, left_eye):
        img = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (const.EYEIMG_SIZE, const.EYEIMG_SIZE))
        img = img.astype('float32')
        img /= 255
        img = img.reshape(1, const.EYEIMG_SIZE, const.EYEIMG_SIZE, 1)
        prediction = self.model.predict(img)
        le_predicted_label = const.EYESTATUS_MAP[np.argmax(prediction[0][:2])]
        return le_predicted_label
    
    
    
    def predict_right_eye_status(self, right_eye):
        img = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (const.EYEIMG_SIZE, const.EYEIMG_SIZE))
        img = img.astype('float32')
        img /= 255
        img = img.reshape(1, const.EYEIMG_SIZE, const.EYEIMG_SIZE, 1)
        prediction = self.model.predict(img)
        re_predicted_label = const.EYESTATUS_MAP[np.argmax(prediction[0][2:])+2]
        return re_predicted_label
        
    