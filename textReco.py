# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 11:33:01 2021

@author: MinhPhuc
"""

import pytesseract
import cv2
pytesseract.pytesseract.tesseract_cmd=r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def preprocess_img(cv2_img):
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    cv2_img = cv2.threshold(cv2_img,200,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2_img = cv2.GaussianBlur(cv2_img,(3,5),0)
    return cv2_img

def img_to_str(img_path):
    img = cv2.imread(img_path)
    img = preprocess_img(img)
    cv2.imshow('result', img)
    result_string = pytesseract.image_to_string(img).replace('\x0c', '')
    return result_string

