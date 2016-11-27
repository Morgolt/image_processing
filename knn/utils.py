import csv
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from knn.core import KNearestNeighbors


def create_test_model(fontfile, digitheight=30):
    for n in range(10):
        # region basic digit
        ttfont = ImageFont.truetype(fontfile, digitheight + 9)
        pil_im = Image.new('RGB', (20, digitheight))
        draw = ImageDraw.Draw(pil_im)
        draw.text((-1, -1), str(n), font=ttfont, fill=255)
        gray = cv2.cvtColor(np.array(pil_im), cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
        cv2.imwrite('res/trainset/%d/bt_2.jpg' % n, thresh)
        # endregion
        # region add gaussian noise
        gaussian_noise = thresh.copy()
        cv2.randn(gaussian_noise, (0), (10))
        cv2.imwrite('res/trainset/%d/bg_2.jpg' % n, thresh + gaussian_noise)
        # endregion
        # region salt&pepper noise
        s_vs_p = 0.5
        amount = 0.2
        out = np.copy(thresh)
        # Salt mode
        num_salt = np.ceil(amount * thresh.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in thresh.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * thresh.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in thresh.shape]
        out[coords] = 0
        cv2.imwrite('res/trainset/%d/bsp_2.jpg' % n, out)
        # endregion


def train_knn():
    model = KNearestNeighbors(read_from_csv())
    return model


def preprocess_train():
    samples = np.empty((0, 30 * 20))
    responses = []
    train_set_directory = 'res/trainset/'
    for class_dir in os.listdir(train_set_directory):
        path = os.path.join(train_set_directory, class_dir)
        for filename in os.listdir(path):
            symbol = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
            _, symbol = cv2.threshold(symbol, 200, 255, cv2.THRESH_BINARY)
            roi = symbol.reshape((1, symbol.shape[0] * symbol.shape[1]))
            samples = np.append(samples, roi, 0)
            responses.append(class_dir)
    samples = np.array(samples, np.short)
    responses = np.array(responses, np.str_)
    return list(zip(samples, responses))


def read_from_csv(path='res/train.csv'):
    result = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            l = list(row)
            result.append(([int(x) for x in l[0:len(l) - 1]], l[-1]))
    return result


def write_to_csv(path='res/train.csv'):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        train = preprocess_train()
        for instance in train:
            writer.writerow([*instance[0], instance[1]])


#if __name__ == '__main__':
#    create_test_model('res/fonts/OcrB Regular.ttf')
#    train_knn()
