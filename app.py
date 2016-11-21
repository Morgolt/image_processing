from uuid import uuid1

import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

import knn


# p, q - точки прямой
# r, s - направляющие векторы
def intersection_point(p, q, r, s):
    if np.isclose(np.cross(r, s), 0):
        return None

    t = np.cross(q - p, s) / np.cross(r, s)
    return p + t * r


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


def reducable_to_rect(contour):
    peri = cv2.contourArea(contour)
    reduced = cv2.approxPolyDP(contour, 0.001 * peri, True)
    # print('reduced contour len:', len(reduced))
    return len(reduced) == 4


def detect_card_number(fname):
    original = cv2.imread(fname)
    original = imutils.resize(original, width=600)
    image_area = np.prod(original.shape[:2])

    image = cv2.GaussianBlur(original, (5, 5), sigmaX=1, sigmaY=1)
    gradient = get_gradient(image, thresh=65)
    # gradient = cv2.Canny(image, 5, 10)

    cts = cv2.findContours(gradient.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    cts.sort(key=cv2.contourArea, reverse=True)
    contour, contour_area = cts[0], cv2.contourArea(cts[0])
    # print('contour_area:', contour_area, '/', image_area, '==', contour_area / image_area)

    if contour_area / image_area < 0.5 or (contour_area / image_area >= 0.4 and not reducable_to_rect(contour)):
        thresh = 65
        while thresh > 15:
            thresh -= 5
            gradient = get_gradient(image, thresh)
            # cv2.imshow('gradient', gradient)
            # cv2.waitKey(0)
            cts = cv2.findContours(gradient.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
            cts.sort(key=cv2.contourArea, reverse=True)
            contour, contour_area = cts[0], cv2.contourArea(cts[0])
            # print('contour_area:', contour_area, '/', image_area, '==', contour_area / image_area)
            # print('thresh:', thresh)
            if contour_area / image_area >= 0.5 or (contour_area / image_area >= 0.4 and reducable_to_rect(contour)):
                break
        else:
            # print('the credit card is too small || couldn\'t compute the card external contour')
            exit(1)

    # print(contour.shape, '-> ...')
    peri, coeff = cv2.arcLength(contour, True), 0.01
    contour = cv2.approxPolyDP(cts[0], coeff * peri, True)
    while contour.shape[0] < 60:
        coeff /= 1.5
        # print('coeff:', coeff)
        contour = cv2.approxPolyDP(cts[0], coeff * peri, True)
        # else:
        # print(contour.shape)

    only_contour = np.zeros_like(image)
    cv2.drawContours(only_contour, [contour], -1, (25, 255, 225), 1)
    # cv2.imshow('only_contour', only_contour)
    # cv2.waitKey(0)

    axis_y = [np.count_nonzero(line) for line in only_contour]
    axis_x = [np.count_nonzero(line) for line in cv2.transpose(only_contour)]
    axis_y_maxs = signal.argrelextrema(np.array(axis_y), lambda a, b: (a >= b) & (a > 20), axis=0, order=10)[0]
    axis_x_maxs = signal.argrelextrema(np.array(axis_x), lambda a, b: (a >= b) & (a > 20), axis=0, order=10)[0]
    # plt.figure()
    # plt.plot(range(1, len(axis_x)+1), axis_x, 'r')
    # plt.plot(range(1, len(axis_y) + 1), axis_y, 'g')
    # plt.scatter(axis_x_maxs, [axis_x[n] for n in axis_x_maxs], c='r')
    # plt.scatter(axis_y_maxs, [axis_y[n] for n in axis_y_maxs], c='g')
    # plt.show()

    contour = contour.reshape((contour.shape[0], 2))
    # left
    max_area = (axis_x_maxs[0] - 10, axis_x_maxs[0] + 10)
    points = contour[(max_area[0] <= contour[:, 0]) & (contour[:, 0] <= max_area[1])]
    line_l = cv2.fitLine(points, distType=cv2.DIST_L1, param=0, reps=0.01, aeps=0.01).T.squeeze()
    # line_ends = ((int(line_l[2] - 1000 * line_l[0]), int(line_l[3] - 1000 * line_l[1])),
    # 			 (int(line_l[2] + 1000 * line_l[0]), int(line_l[3] + 1000 * line_l[1])))
    # cv2.line(only_contour, line_ends[0], line_ends[1], (0, 0, 255))
    # print(line_l, points.shape)

    # right
    max_area = (axis_x_maxs[-1] - 10, axis_x_maxs[-1] + 10)
    points = contour[(max_area[0] <= contour[:, 0]) & (contour[:, 0] <= max_area[1])]
    line_r = cv2.fitLine(points, distType=cv2.DIST_L1, param=0, reps=0.01, aeps=0.01).T.squeeze()
    # line_ends = ((int(line_r[2] - 1000 * line_r[0]), int(line_r[3] - 1000 * line_r[1])),
    # 			 (int(line_r[2] + 1000 * line_r[0]), int(line_r[3] + 1000 * line_r[1])))
    # cv2.line(only_contour, line_ends[0], line_ends[1], (0, 0, 255))
    # print(line_r, points.shape)

    # top
    max_area = (axis_y_maxs[0] - 10, axis_y_maxs[0] + 10)
    points = contour[(max_area[0] <= contour[:, 1]) & (contour[:, 1] <= max_area[1])]
    line_t = cv2.fitLine(points, distType=cv2.DIST_L1, param=0, reps=0.01, aeps=0.01).T.squeeze()
    # line_ends = ((int(line_t[2] - 1000 * line_t[0]), int(line_t[3] - 1000 * line_t[1])),
    # 			 (int(line_t[2] + 1000 * line_t[0]), int(line_t[3] + 1000 * line_t[1])))
    # cv2.line(only_contour, line_ends[0], line_ends[1], (0, 0, 255))
    # print(line_t, points.shape)

    # bottom
    max_area = (axis_y_maxs[-1] - 10, axis_y_maxs[-1] + 10)
    points = contour[(max_area[0] <= contour[:, 1]) & (contour[:, 1] <= max_area[1])]
    line_b = cv2.fitLine(points, distType=cv2.DIST_L1, param=0, reps=0.01, aeps=0.01).T.squeeze()
    # line_ends = ((int(line_b[2] - 1000 * line_b[0]), int(line_b[3] - 1000 * line_b[1])),
    # 			 (int(line_b[2] + 1000 * line_b[0]), int(line_b[3] + 1000 * line_b[1])))
    # cv2.line(only_contour, line_ends[0], line_ends[1], (0, 0, 255))
    # print(line_b, points.shape)

    point_tl = intersection_point(line_l[2:4], line_t[2:4], line_l[:2], line_t[:2])
    point_tr = intersection_point(line_r[2:4], line_t[2:4], line_r[:2], line_t[:2])
    point_bl = intersection_point(line_l[2:4], line_b[2:4], line_l[:2], line_b[:2])
    point_br = intersection_point(line_r[2:4], line_b[2:4], line_r[:2], line_b[:2])

    points = np.array([point_tl, point_tr, point_br, point_bl])
    # for point in points:
    # 	cv2.circle(only_contour, tuple(point), 5, (255,255,25), lineType=-1)

    card_height = 250
    width1 = np.sqrt(((point_br[0] - point_bl[0]) ** 2) + ((point_br[1] - point_bl[1]) ** 2))
    width2 = np.sqrt(((point_tr[0] - point_tl[0]) ** 2) + ((point_tr[1] - point_tl[1]) ** 2))
    height1 = np.sqrt(((point_tr[0] - point_br[0]) ** 2) + ((point_tr[1] - point_br[1]) ** 2))
    height2 = np.sqrt(((point_tl[0] - point_bl[0]) ** 2) + ((point_tl[1] - point_bl[1]) ** 2))
    # print(width1, width2, height1, height2)
    card_width = int(max(width1, width2) * card_height / max(height1, height2))
    # print(card_width, card_height)

    warped = np.array([[0, 0], [card_width - 1, 0], [card_width - 1, card_height - 1], [0, card_height - 1]],
                      dtype="float32")
    M = cv2.getPerspectiveTransform(points, warped)
    warped = cv2.warpPerspective(original, M, (card_width, card_height))

    card_number = warped[138:164, :]
    return card_number
    # cv2.imshow('warped', warped)
    # cv2.imshow('card_number', card_number)
    # cv2.waitKey(0)


def detect_single_digits(edged_number):
    ymax = np.flatnonzero(edged_number.max(axis=0))
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
        digits.append(edged_number[:, inds[0]:inds[1] + 1])
    for digit in digits:
        cv2.imshow('first digit', digit)
        cv2.waitKey(0)
    return digits


def projection_x(img, y0, h_l, show_plot=False):
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


def segment_digits(contrs, show_img=False):
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
    if show_img:
        for digit in digits:
            roi = cv2.resize(digit, (20, 30))
            cv2.imwrite('res/temp/%d.jpg' % uuid1(), roi)
            i += 1
    return digits


def recognize_block(block, show_img=False):
    dsize = (90, 30)
    block = cv2.resize(block, dsize)
    block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
    block = cv2.GaussianBlur(block, (3, 3), sigmaX=1, sigmaY=1)
    contrs = get_gradient(block, thresh=100)
    kernel = np.ones((3, 1), np.uint8)
    contrs = cv2.morphologyEx(contrs, cv2.MORPH_CLOSE, kernel)
    digits_block = segment_digits(contrs)
    block = []
    for digit in digits_block:
        roi = cv2.resize(digit, (20, 30))
        if show_img:
            cv2.imshow('thresh_block', roi)
            cv2.waitKey(0)
        roi = np.float32(roi.reshape((1, 30 * 20)))
        # ret, result, neighbours, dist = cv2.ml.model.findNearest(roi, 3)
        ret = knn.model.classify(roi)
        block.append(int(ret))

    return block


def get_printable_number(number):
    a = [str(x) for x in number]
    a = [a[x:x + 4] for x in range(0, 16, 4)]
    result = 'Номер карты:\n{a[0]} {a[1]} {a[2]} {a[3]}'.format(a=[''.join(d) for d in a])
    return result


def main(fname):
    card_number = detect_card_number(fname)
    dsize = (400, 30)
    card_number = cv2.resize(card_number, dsize)
    gray = cv2.cvtColor(card_number, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    y0 = np.arange(15, 40, 1)
    h_l = np.arange(70, 90, 1)
    inds = projection_x(blur, y0, h_l, False)
    numbers = []
    for i in range(0, 4):
        numbers.extend(recognize_block(card_number[:, inds[i]:inds[i + 1]], False))
    print(get_printable_number(numbers))


main('res/test%d.jpg' % 1)
