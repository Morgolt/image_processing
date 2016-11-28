import cv2
import numpy as np

import knn
from utils import show_image, detect_single_symbols


def recognize_owner(owner):
    owner_gray = cv2.cvtColor(owner, cv2.COLOR_BGR2GRAY)
    _, owner_gradient = cv2.threshold(owner_gray, 150, 255, cv2.THRESH_BINARY)

    symbols = detect_single_symbols(owner_gradient, True)
    for symbol in symbols:
        # find countour of letter
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        digit_closed = cv2.morphologyEx(symbol, cv2.MORPH_CLOSE, kernel)
        cts = cv2.findContours(digit_closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        cts.sort(key=cv2.contourArea, reverse=True)
        digit_countour = cts[0]
        digit_rect = cv2.boundingRect(digit_countour)
        digit_base = symbol[digit_rect[1]:digit_rect[1] + digit_rect[3], :]
        roi = cv2.resize(digit_base, (20, 30))
        # show_image(roi)

        roi = np.float32(roi.reshape((1, 30 * 20)))
        ret = knn.symbol_model.classify(roi, 5)
        print(str(ret)[2], end="")
    return symbols
