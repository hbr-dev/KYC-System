#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 18:37:26 2024

@author: kudu
"""


from utilities.helper import Helper

import cv2
import numpy as np





class ImageProcessingUtils:

    def __init__(self):
        self.helper_instance = Helper()
        pass





    def convert_2_gray(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray_image
    
    
    
    def convert_2_rgb(self, bgr_image):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return rgb_image
    
    
    
    def reduce_shine(self, gray_image):
        # Apply histogram equalization to reduce shine
        equalized_image = cv2.equalizeHist(gray_image)
        return equalized_image
    
    
    
    def reduce_brightness(self, image, factor):
        # Convert the image to a float32 array
        image_float = image.astype(np.float32)
        # Reduce the brightness by multiplying pixel values with a factor
        darkened_image = np.clip(image_float * factor, 0, 255).astype(np.uint8)
        return darkened_image
    
    
    
    def binarization(self, image):
        img, thresh = cv2.threshold(image, 0,255, cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
        return img, thresh
    
    
    
    def invert_colors(self, optical_chars):
        chars = []
        for optical_char in optical_chars:
            optical_char = cv2.bitwise_not(optical_char)
            gray_color = (211, 211, 211)  # RGB value for gray

            # Get the image dimensions
            height, width, channels = optical_char.shape

            # Iterate through each pixel and change the white background to gray
            for y in range(height):
                for x in range(width):
                    if all(optical_char[y, x] == [255, 255, 255]):
                        optical_char[y, x] = gray_color
            chars.append(optical_char)
        return chars
    
    
    
    def zoom_in(self, image, x, y, width, height, scale):
        # Define the region of interest (ROI) based on the provided coordinates
        roi = image[y:y+height, x:x+width]
        # Resize the ROI to zoom in
        zoomed_roi = cv2.resize(roi, (int(width*scale), int(height*scale)))
        # Replace the original ROI with the zoomed-in version
        image[y:y+int(height*scale), x:x+int(width*scale)] = zoomed_roi
        return image
    
    
    
    def dilate(self, image, words= False):
        img = image.copy()
        m = 3
        n = m - 2                   # n less than m for Vertical structuring element to dilate chars
        itrs = 4
        if words:
            m = 6
            n = m
            itrs = 3
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n, m))
        dilation = cv2.dilate(img, rect_kernel, iterations = itrs)
        return dilation
    
    
    
    def uc_roi(self, image):
        height, width = image.shape # The passport model dim: 661X1000
        x = 15
        y = round ((height*5)/7)
        end_x = round((width*2)/3)
        end_y = height - 15
        
        roi = image[y:end_y, x:end_x]
        return roi
    
    
    
    def oc_extractor(self, image):
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        # Filter and sort contours
        filtered_contours = []
        contours_rects = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 10:
                filtered_contours.append(cnt)
                contours_rects.append((x, y, w, h, cnt))
                
        # Sort contours based on x-coordinate and then y-coordinate
        contours = self.helper_instance.sort_contours(contours_rects)
        
        char_images = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            char_img = image[y:y+h, x:x+w]
            char_images.append(char_img)
        return char_images
            
    
        
    def resize_and_pad_images(self, binarized_images):
        resized_images = []
        
        for img in binarized_images:
            resized_img = cv2.resize(img, (16, 16), interpolation=cv2.INTER_NEAREST)
            
            # Determine the number of channels
            if len(resized_img.shape) == 2:
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
            elif len(resized_img.shape) == 3:
                pass
            else:
                raise ValueError("Invalid image shape: {}".format(resized_img.shape))
            
            # Pad the image to 32x32 with 3 channels
            padded_img = np.zeros((32, 32, 3), dtype=np.uint8)
            h, w, _ = resized_img.shape
            x_offset = (32 - w) // 2
            y_offset = (32 - h) // 2
            padded_img[y_offset:y_offset+h, x_offset:x_offset+w, :] = resized_img
            
            resized_images.append(padded_img)
        
        return resized_images
    
    
    




