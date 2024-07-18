#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:59:41 2024

@author: kudu
"""

from utilities.image_io_utils import ImageIOUtils
from utilities.image_processing_utils import ImageProcessingUtils
from utilities.ocr_model_utils import OCRModelUtils






    
image_io_manager = ImageIOUtils()
image_processing_manager = ImageProcessingUtils()

image = image_io_manager.read_image('resources/passportTN.png')

image = image_io_manager.normalize_image_size(image)

image = image_processing_manager.reduce_brightness(image, .5)
image = image_processing_manager.convert_2_gray(image)
_,image = image_processing_manager.binarization(image)

image = image_processing_manager.uc_roi(image)
optical_chars = image_processing_manager.oc_extractor(image)
optical_chars = image_processing_manager.resize_and_pad_images(optical_chars)

ocr_model_utils = OCRModelUtils("models/ocr_model.keras")

inverted_oc = image_processing_manager.invert_colors(optical_chars)

print("Full Name: " + ocr_model_utils.predict_fullName(inverted_oc))

# image_io_manager.display_images(optical_chars)
# image_io_manager.display_image(inverted_color, "Chars")
# first_name = surname = اللقب
# last_name = given_name = الإسم


