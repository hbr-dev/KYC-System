#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 18:36:49 2024

@author: kudu
"""

import cv2
import numpy as np





class ImageIOUtils:

    def __init__(self):
        pass





    def read_image(self, image_path):
        # Read an image from the specified path
        image = cv2.imread(image_path)
        return image
    
    
    
    def normalize_image_size(self, image):
        # Resize the image to the standard size
        resized_image = cv2.resize(image, (600, 445), interpolation=cv2.INTER_CUBIC)
        return resized_image
    
    
    
    def display_image(self, image, title='Image'):
        # Display the image in a window
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    
    
    def display_images(self, images, title='Multi-img display', cols=8, rows=8, img_H=32, img_W=32):
        # Create a single window to display the images
        fig = np.zeros((rows * img_H, cols * img_W, 3), dtype=np.uint8)

        # Populate the window with the images
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < len(images):
                    fig[i*img_H:(i+1)*img_H, j*img_W:(j+1)*img_W, :] = images[idx]

        # Display the window
        cv2.imshow(title, fig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    def close_image(self):
        # Close all OpenCV windows
        cv2.destroyAllWindows()
    
    
    
    def save_image(self, image, save_path):
        # Save the image to the specified path
        cv2.imwrite(save_path, image)
