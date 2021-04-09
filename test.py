# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 00:15:04 2021

@author: MinhPhuc
"""

# import cv2
# import pytesseract

# # set the path to use tesseract
# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# # show the image
# img = cv2.imread('demo.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # you have to convert your img to rgb to show it
# print(pytesseract.image_to_string(img))# print the text inside the img
# cv2.imshow('result', img)
# cv2.waitKey(0)


import cv2
import pytesseract
import imutils

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image = cv2.imread('demo2.jpg')
image = imutils.resize(image, width=700)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
thresh = cv2.GaussianBlur(thresh, (3,3), 0)
data = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6')
print(data[:-5])

cv2.imshow('thresh', thresh)
cv2.imwrite('thresh.png', thresh)
cv2.waitKey()