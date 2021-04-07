# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 23:22:31 2021

@author: MinhPhuc
"""

import pytesseract
import cv2
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd=r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

img = cv2.imread('alto.png')
img = cv2.threshold(img,200,255, cv2.THRESH_BINARY_INV)[1]

img = cv2.resize(img,(0,0),fx=3,fy=3)
img = cv2.GaussianBlur(img,(11,11),0)
img = cv2.medianBlur(img,9)
cv2.imshow('asd',img)


img2char = pytesseract.image_to_string(img,  lang='eng', config='--psm 7')
print(repr(img2char))
cv2.waitKey(0)


