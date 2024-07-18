#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:24:53 2024

@author: kudu
"""

from utilities.image_processing_utils import ImageProcessingUtils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, Flatten, Dense, Dropout

# import matplotlib.pyplot as plt
import numpy as np

import cv2
import sys





class FaceMatchingUtils:
    
    def __init__(self):
        self.image_processing_manager = ImageProcessingUtils()
        self.face_detector            = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        self.video_capture            = cv2.VideoCapture(0)
    
    
    
    def detect_faces(self, gray_image):
        faces = self.face_detector.detectMultiScale(
            gray_image,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
        
        
        
    def face_matching_rt(self):
        try:
            print("Camera device released")
            self.__center_window()
            while True:
                frame = self.detect_and_match()
                cv2.imshow("Faces Matching", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print("An exception occurred:", e)
            filename = sys.exc_info()[2].tb_frame.f_code.co_filename
            line_number = sys.exc_info()[2].tb_lineno
            print("File:", filename)
            print("Line:", line_number)

        finally:
            cv2.destroyAllWindows()
            self.video_capture.release()
            print("Camera device released")
        
        
        
    def detect_and_match(self):
        _, frame = self.video_capture.read()
        frame    = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
        frame    = cv2.flip(frame, 1)
        
        gray = self.image_processing_manager.convert_2_gray(frame)
        
        faces = self.detect_faces(gray)
        
        color = (0, 0, 255)
        
        if len(faces) == 2:
            # Extract the two largest faces
            face1 = frame[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]
            face2 = frame[faces[1][1]:faces[1][1]+faces[1][3], faces[1][0]:faces[1][0]+faces[1][2]]

            # Compare the two faces
            match_score = self.__match_faces(face1, face2)
            
            if match_score > .5:
                color = (0, 255, 0)

            # Draw the match score on the frame
            cv2.putText(frame, f"Match score: {match_score * 100:.2f} %", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(frame, "...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        return frame
    
    
    
    def __preprocess_face(self, face):
        img = face
        img = cv2.resize(face, (100, 100))
        img = img.astype('float32')
        img /= 255
        return img
    
    
    
    def __match_faces(self, face1, face2, threshold=.5):
        
        # Resize the faces to a common size
        face1 = cv2.resize(face1, (100, 100))
        face2 = cv2.resize(face2, (100, 100))

        # Convert the faces to grayscale and calculate the difference
        face1_gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
        face2_gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
        
        ssim = cv2.matchTemplate(face1_gray, face2_gray, cv2.TM_CCOEFF_NORMED)
        match_score = ssim[0, 0]

        return match_score
    
    
    
    """
    def __create_facenet_model(self):
        model = Sequential()
        model.add(Conv2D(64, (7, 7), strides=(2, 2), padding='same', input_shape=(224, 224, 3)))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(ReLU())
    
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(ReLU())
    
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(128))
        
        model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=['accuracy'])
        return model
    """
    
    
    
    def __center_window(self):
        # Get the screen resolution
        screen_width = 1920
        screen_height = 1080
        
        # Set the window size to 60% of the screen size
        window_width = int(screen_width * 0.6)
        window_height = int(screen_height * 0.6)
        
        # Calculate the position to center the window
        x = int((screen_width - window_width) / 2) + 200
        y = int((screen_height - window_height) / 2)
        
        # Create the window and set its position
        cv2.namedWindow("Faces Matching")
        cv2.moveWindow("Faces Matching", x, y)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    