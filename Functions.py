# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 09:28:35 2026

@author: aprash
"""

import matplotlib.pyplot as plt
import cv2

def show_img(path):
    
    image = cv2.imread(path)    
    plt.imshow(image)
    plt.axis('off')
    plt.show()