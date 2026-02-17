# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 09:28:35 2026

@author: aprash
"""

import matplotlib.pyplot as plt
import cv2

def show_img(path, x=20, y=12):
    image = cv2.imread(path)
    
    if image is None:
        print(f"Error: Could not load image at {path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(x, y)) 
    
    plt.imshow(image)
    plt.axis('off')
    plt.show()