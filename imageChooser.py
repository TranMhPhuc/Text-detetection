# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 08:36:02 2021

@author: MinhPhuc
"""

from tkinter import *
from tkinter.filedialog import askopenfile 
from PIL import ImageTk, Image

import cv2
import textDetect
import textReco

root = Tk();
root.title("Text detection!!!")
root.geometry('1100x700')
root.iconbitmap('favicon.ico')
root.resizable(width=0, height=0)

# ========================================================================================

# ham xu ly cho giao dien
    # lay du lieu tu textbox
def retrieve_input():
    global path_entry
    inputValue=path_entry.get()
    return inputValue

    # set duong dan sau khi chon anh xong
def set_path(text):
    global path_entry
    path_entry.config(state='normal')
    path_entry.delete(0, END)
    path_entry.insert(0, text)
    path_entry.configure(state='disabled')

    # xuat hien anh len man hinh
def show_img():
    global img_label
    global img
    img_info = retrieve_input()
    img = ImageTk.PhotoImage(Image.open(img_info).resize((640, 320)))
    img_label.grid_forget()
    img_label = Label(image=img)
    img_label.grid(row=2, column=1)
    return


    # xuat hien dialog de chon anh
def open_file(): 
    file = askopenfile(mode ='r', filetypes =[('Image Files', '*.jpg *.png *.jpeg')]) 
    if file is not None: 
        set_path(file.name)
        show_img()


    # ham nay dung de xu ly anh
    # ham nhan dang
def text_recognize():
    global text_textbox
    img_info = retrieve_input()
    result_text = textReco.img_to_str(img_info)
    text_textbox.delete('1.0', 'end')
    text_textbox.insert('1.0', result_text)
        
    
    #ham nhan biet chu va dong khung chu

def word_detection():
    global word_textbox
    img_info = retrieve_input()    
    words, result_img = textDetect.text_detector(img_info)
    cv2.imshow('result', result_img)
    word_textbox.delete('1.0', 'end')
    word_textbox.insert('1.0', ''.join(words))
    
# ========================================================================================
# thiet ke giao dien
    # nut chon anh
choose_img_btn = Button(root, text ='Chon anh', command = lambda:open_file(), bg='#0ea7c2')
choose_img_btn.grid(row=0, column=2)

    # khung text box hien thi duong dan anh
path_entry = Entry(root, width=60)
set_path('demo.jpg')
path_entry.grid(row=0, column=1)


    # noi xuat hinh anh len man hinh
img = ImageTk.PhotoImage(Image.open('demo.jpg').resize((640, 320)))
img_label = Label(image=img)
img_label.grid(row=2, column=1)

    # nut bam de xu ly anh
        #nut de lay text ra
text_recognize_btn = Button(root, text ='Nhan dien van ban!', command = lambda:text_recognize(), bg='#34db21')
text_recognize_btn.grid(row=3, column=1)
        #nut de dong khung text
text_detection_btn = Button(root, text ='Nhan biet chu!', command = lambda:word_detection(), bg='#c7db60')
text_detection_btn.grid(row=3, column=2)
    
    # textbox hien ket qua lay text tu anh
text_textbox = Text(root,height=15, width=60, font=("Helvetica", 12))
text_textbox.grid(row=4, column=1)

word_textbox = Text(root,height=15, width=40, font=("Helvetica", 12))
word_textbox.grid(row=4, column=2)
    
    # nut thoat chuong trinh
quit_button= Button(root, text='Thoat chuong trinh', command=root.destroy, bg='#f5424b')
quit_button.grid(row=5, column=2)

root.mainloop()