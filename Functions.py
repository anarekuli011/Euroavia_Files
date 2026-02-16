# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 09:28:35 2026

@author: aprash
"""

import matplotlib.pyplot as plt
import cv2

def show_img(path, zoom_factor=1.0):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if zoom_factor < 1.0:
        h, w, _ = image.shape
        dy = int(h * (1 - zoom_factor) / 2)
        dx = int(w * (1 - zoom_factor) / 2)
        
        image = image[dy:h-dy, dx:w-dx]

    plt.imshow(image)
    plt.axis('off')
    plt.show()