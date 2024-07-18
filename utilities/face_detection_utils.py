#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:29:43 2024

@author: kudu
"""

from utilities.liveness_recognition_model_utils import LivenessRecognitionModelUtils
from utilities.image_processing_utils import ImageProcessingUtils
from utilities.image_io_utils import ImageIOUtils

from collections import defaultdict

# import matplotlib.pyplot as plt

import cv2
import sys





class FaceDetectionUtils:
    
    def __init__(self):
        self.face_detector            = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        self.open_eyes_detector       = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        self.left_eye_detector        = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
        self.right_eye_detector       = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
        self.video_capture            = cv2.VideoCapture(0)
        self.image_processing_manager = ImageProcessingUtils()
        self.image_io_manager         = ImageIOUtils()
        self.trained_model_utils      = LivenessRecognitionModelUtils('models/aliveness_model.keras')
    
    
    
    def detect_faces(self, gray_image):
        faces = self.face_detector.detectMultiScale(
            gray_image,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
    
    
    
    def detect_eyes_with_glasses(self, gray_face):
        open_eyes_glasses = self.open_eyes_detector.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        return open_eyes_glasses
    
    
    
    def detect_left_eye(self, left_face_gray):
        left_eye = self.left_eye_detector.detectMultiScale(
            left_face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        return left_eye
    
    
    
    def detect_right_eye(self, right_face_gray):
        right_eye = self.right_eye_detector.detectMultiScale(
            right_face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        return right_eye
        
        
        
    def liveness_detection_rt(self):
        try:
            print("Camera device released")
            eyes_detected = defaultdict(str)
            verified_times = 0
            self.__center_window()
            while True:
                frame, verified_times = self.detect_and_display(eyes_detected, verified_times)
                cv2.imshow("Face Liveness Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if verified_times >= 3:
                    print("It seams like you are a real person")
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
        
        
        
    def detect_and_display(self, eyes_detected, verified_times):
        _, frame = self.video_capture.read()
        frame    = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
        frame    = cv2.flip(frame, 1)
        
        gray = self.image_processing_manager.convert_2_gray(frame)
        
        faces = self.detect_faces(gray)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h,x:x+w]
            gray_face = gray[y:y+h,x:x+w]
            
            """if len(self.detect_eyes_with_glasses(gray_face)) == 2:
                eyes_detected["user"]+='1'
                for (ex,ey,ew,eh) in self.detect_eyes_with_glasses(gray_face):
                    cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)   
            else:"""
            left_face = frame[y:y+h, x+int(w/2):x+w]
            left_face_gray = gray[y:y+h, x+int(w/2):x+w]
            right_face = frame[y:y+h, x:x+int(w/2)]
            right_face_gray = gray[y:y+h, x:x+int(w/2)]
            # we suppose the eyes are open
            eye_status = '1'
            
            # For each eye check wether the eye is closed.
            # If one is closed we conclude the eyes are closed
            re_status = self.__right_eye_status(right_face_gray, right_face)
            le_status = self.__left_eye_status(left_face_gray, left_face)
    
            eye_status = str(re_status + le_status - 1) if re_status + le_status > 1 else str(re_status + le_status)
            eyes_detected["user"] += eye_status
    
            if self.__isBlinking(eyes_detected["user"],4):
                print(eyes_detected)
                verified_times += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                eyes_detected["user"] = ''
                y = y - 15 if y - 15 > 15 else y + 15
                cv2.putText(frame, str(verified_times), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        return frame, verified_times

    
    
    def __right_eye_status(self, right_face_gray, right_face):
        eye_status = 1
        color = (0,255,0)
        for (ex,ey,ew,eh) in self.detect_right_eye(right_face_gray):
            eye_status = self.trained_model_utils.predict_eye_status(right_face[ey:ey+eh,ex:ex+ew])
            cv2.rectangle(right_face,(ex,ey),(ex+ew,ey+eh),color,2)
            if eye_status == 0:
                color = (0,0,255)
            cv2.rectangle(right_face,(ex,ey),(ex+ew,ey+eh),color,2)
        return eye_status
    
    
    
    def __left_eye_status(self, left_face_gray, left_face):
        eye_status = 1
        color = (0,255,0)
        for (ex,ey,ew,eh) in self.detect_left_eye(left_face_gray):
            eye_status = self.trained_model_utils.predict_eye_status(left_face[ey:ey+eh,ex:ex+ew])
            cv2.rectangle(left_face,(ex,ey),(ex+ew,ey+eh),color,2)
            if eye_status == 0:
                color = (0,0,255)
            cv2.rectangle(left_face,(ex,ey),(ex+ew,ey+eh),color,2)
        return eye_status
    
    
    
    def __isBlinking(self, history, maxFrames):
        """
        @history: A string containing the history of eyes status 
        where a '1' means that the eyes were closed and '0' open.
        @maxFrames: The maximal number of successive frames where an eye is closed
        """
        for i in range(maxFrames):
            pattern = '1' + '0'*(i+1) + '1'
            if pattern in history:
                return True
        return False
    
    
    
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
        cv2.namedWindow("Face Liveness Detector")
        cv2.moveWindow("Face Liveness Detector", x, y)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    