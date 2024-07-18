#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 18:49:47 2024

@author: kudu
"""

from utilities.model_utils import ModelUtils
from constants import const
from itertools import takewhile

# import matplotlib.pyplot as plt
import numpy as np

import cv2





class OCRModelUtils(ModelUtils):
    
    def __init__(self, model_path):
        super().__init__(model_path)
    
    
    
    
    
    def predict_fullName(self, optical_chars):
        recognized_str = self.__recognized_str(optical_chars)
        spr_matched_values = self.match_optical_characters(optical_chars[5:])
        first_name, index = self.__copy_until_threshold(recognized_str, spr_matched_values)
        remaining_chars = recognized_str[index + 2:]
        last_name, _ = self.__copy_until_threshold(remaining_chars, spr_matched_values[index + 2:])
        return first_name + " " + last_name
    
    
    
    def match_optical_characters(self, optical_chars):
        ref_img = cv2.imread("resources/ref.jpg")

        matching_values = []
        for optical_char in optical_chars:
            score = cv2.matchTemplate(ref_img, optical_char, cv2.TM_CCOEFF_NORMED)
            matching_values.append(score)
        
        scores = [score[0][0] for score in matching_values]
        
        return scores
       
    
    
    
    def __recognized_str(self, optical_chars):
        """
        fig, axes = plt.subplots(6, 6, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            image = optical_chars[i]
            prediction = self.model.predict(np.array([image]))
            predicted_label = const.ALPHA_MAP[np.argmax(prediction)]
            ax.imshow(image, cmap='gray')
            ax.set_title(f"Predicted: {predicted_label}: {np.argmax(prediction)}%")
            ax.axis('off')
        plt.show()
        """
        recognized_str = ''
        for optical_char in optical_chars:
             prediction = self.model.predict(np.array([optical_char]))
             recognized_str += const.ALPHA_MAP[np.argmax(prediction)]
        recognized_str = recognized_str[5:]
        return recognized_str
    
    
    
    def __copy_until_threshold(self, input_string, threshold_list):
        output_str = ""
        for i in range(len(input_string)):
            if threshold_list[i] >= .7:
                return output_str, i
            else:
                output_str += input_string[i]
        





