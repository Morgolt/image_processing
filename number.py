import cv2
import numpy as np
from matplotlib import pyplot as plt

import knn
import utils


def recognize_number(number_image, show_img=True):
    dsize = (400, 30)
    card_number = cv2.resize(number_image, dsize)
    gray = cv2.cvtColor(card_number, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    y0 = np.arange(15, 40, 1)
    h_l = np.arange(70, 90, 1)
    inds = projection_x(blur, y0, h_l, True)
    numbers = []
    if show_img:
        fig = plt.figure()
        for i in range(0, 4):
            a = fig.add_subplot(1, 4, i + 1)
            image_plot = plt.imshow(card_number[:, inds[i]:inds[i + 1]])
        plt.show()
    for i in range(0, 4):
        numbers.extend(recognize_block(card_number[:, inds[i]:inds[i + 1]], False))

    print(get_printable_number(numbers))


def projection_x(img, y0, h_l, show_plot=True):
    projection = np.sum(img, axis=0)
    Y0 = 0
    HL = 0
    i = np.array([0, 1, 2, 3, 4])
    argmin = np.sum(projection)
    for x in y0:
        for y in h_l:
            inds = np.add(x, np.multiply(y, i))
            summ = np.sum(projection[inds])
            if summ <= argmin:
                argmin = summ
                Y0 = x
                HL = y

    inds = np.add(Y0, np.multiply(HL, i))
    if show_plot:
        plt.plot(projection)
        ymax = np.max(projection)
        for x in inds:
            plt.plot((x, x), (0, ymax), '-r')
        plt.xlim([0, len(projection)])
        plt.show()
    return inds


def recognize_block(block, show_img=False):
    dsize = (90, 30)
    block = cv2.resize(block, dsize)
    block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
    block = cv2.GaussianBlur(block, (3, 3), sigmaX=1, sigmaY=1)
    contrs = utils.get_gradient(block, thresh=100)
    kernel = np.ones((3, 1), np.uint8)
    contrs = cv2.morphologyEx(contrs, cv2.MORPH_CLOSE, kernel)
    digits_block = segment_digits(contrs)
    block = []
    for digit in digits_block:
        roi = cv2.resize(digit, (20, 30))
        roi = np.float32(roi.reshape((1, 30 * 20)))
        ret = knn.digits_model.classify(roi)
        block.append(int(ret))

    return block


def segment_digits(contrs):
    ymax = np.flatnonzero(contrs.max(axis=0))
    digits = []
    digit_indices = []
    i = 0
    index = ymax[0]
    count = 0
    while i < len(ymax) - 1:
        if ymax[i] + 1 == ymax[i + 1]:
            count += 1
        else:
            digit_indices.append((index, ymax[i]))
            index = ymax[i + 1]
            count = 0
        i += 1
    if count:
        digit_indices.append((index, ymax[i]))
    for inds in digit_indices:
        width = inds[1] - inds[0]
        if width < 10:
            continue
        elif width > 17:
            cnt = np.math.floor(width / 17)
            if cnt > 3:
                cnt = 3
            div = [(inds[0] + 17 * (j - 1), 17 * j) for j in range(1, cnt + 2)]
            digits.extend([contrs[:, ind[0]:ind[1] + 1] for ind in div])
        else:
            digits.append(contrs[:, inds[0]:inds[1] + 1])
        if len(digits) == 16:
            break
    return digits


def get_printable_number(number):
    a = [str(x) for x in number]
    a = [a[x:x + 4] for x in range(0, 16, 4)]
    result = 'Номер карты:\n{a[0]} {a[1]} {a[2]} {a[3]}'.format(a=[''.join(d) for d in a])
    return result
