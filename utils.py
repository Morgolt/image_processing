import cv2
import numpy as np


def get_gradient(image, thresh):
    channels = cv2.split(image)
    result = np.zeros_like(channels[0])
    for channel in channels:
        edgedx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        edgedy = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        edged = np.sqrt(edgedx ** 2 + edgedy ** 2)
        edged = edged / edged.max() * 255
        edged = np.uint8(edged)
        _, edged = cv2.threshold(edged, thresh, 255, cv2.THRESH_BINARY)
        result += edged
    return result


def intersection_point(p, q, r, s):
    if np.isclose(np.cross(r, s), 0):
        return None

    t = np.cross(q - p, s) / np.cross(r, s)
    return p + t * r


def reducable_to_rect(contour):
    peri = cv2.contourArea(contour)
    reduced = cv2.approxPolyDP(contour, 0.001 * peri, True)
    return len(reduced) == 4


def show_image(image, image_name="image"):
    cv2.imshow(image_name, image)
    cv2.waitKey(0)


def detect_single_symbols(symbol_img, show_img=False):
    ymax = np.flatnonzero(symbol_img.max(axis=0))
    symbols = []
    symbols_indices = []
    i = 0
    index = ymax[0]
    count = 0
    while i < len(ymax) - 1:
        if ymax[i] + 1 == ymax[i + 1]:
            count += 1
        else:
            symbols_indices.append((index, ymax[i]))
            index = ymax[i + 1]
            count = 0
        i += 1
    if count:
        symbols_indices.append((index, ymax[i]))
    for inds in symbols_indices:
        symbols.append(symbol_img[:, inds[0]:inds[1] + 1])
    if show_img:
        for digit in symbols:
            cv2.imshow('first digit', digit)
            cv2.waitKey(0)
    return symbols
