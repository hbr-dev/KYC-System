#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:54:30 2024

@author: kudu
"""

import tensorflow as tf






class ModelUtils:
    
    
    def __init__(self, model_path):
        try:
            self.model = tf.keras.models.load_model(model_path)
        except OSError as e:
            raise FileNotFoundError(f"Model file not found at path: {model_path}. Error: {e}")
    
    
    
    
    
    def load_model(self):
        return self.model
    
    
    
    