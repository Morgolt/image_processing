import csv
import os
import string

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_trainset(fontfile, digitheight=30, create_thin = True):
    for letter_str in string.ascii_uppercase:

        font_name = os.path.basename(fontfile).split(".")[0].replace(" ", "").lower()

        #create dir for trainset
        directory = 'res/trainset/%s' % letter_str
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        ttfont = ImageFont.truetype(fontfile, 100)
        pil_im = Image.new('RGB', (200, 200))
        draw = ImageDraw.Draw(pil_im)
        draw.text((-1, -1), str(letter_str), font=ttfont, fill=(255, 255, 255))
        gray = cv2.cvtColor(np.array(pil_im), cv2.COLOR_BGR2GRAY)

        cts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        cts.sort(key = cv2.contourArea, reverse = True)
        letter_countur = cts[0]
        letter_rect = cv2.boundingRect(letter_countur)

        #base letter
        letter_base = gray[letter_rect[1]:letter_rect[1]+letter_rect[3], letter_rect[0]:letter_rect[0]+letter_rect[2]]
        letter_base = cv2.resize(letter_base, (20, 30))    
        cv2.imwrite('res/trainset/%s/%s_base.jpg' % (letter_str, font_name), letter_base)

        #base with gaussian noise
        gaussian_noise = letter_base.copy()
        cv2.randn(gaussian_noise, (0), (1))
        letter_base_gaussian_noise = letter_base + gaussian_noise
        cv2.imwrite('res/trainset/%s/%s_base_gaussian_noise.jpg' % (letter_str, font_name), letter_base_gaussian_noise)

        #thin litter
        if create_thin:
            letter_img_thin = cv2.erode(letter_base, None, iterations = 1)
            cv2.imwrite('res/trainset/%s/%s_thin.jpg' % (letter_str, font_name), letter_img_thin)