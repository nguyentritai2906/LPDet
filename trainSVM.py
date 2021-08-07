import glob

import cv2
import numpy as np
from tqdm import tqdm

DIGIT_WIDTH = 30
DIGIT_HEIGHT = 60

write_path = "data/"


def get_digit_data(path):

    digit_list = []
    label_list = []

    for number in tqdm(range(10)):
        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            #  print(img_org_path)
            img = cv2.imread(img_org_path, 0)
            img = np.array(img)
            img = img.reshape(-1, DIGIT_HEIGHT * DIGIT_WIDTH)

            #  print(img.shape)

            digit_list.append(img)
            label_list.append([int(number)])

    for number in tqdm(range(65, 91)):
        #  print(number)
        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            #  print(img_org_path)
            img = cv2.imread(img_org_path, 0)
            img = np.array(img)
            img = img.reshape(-1, DIGIT_HEIGHT * DIGIT_WIDTH)

            #  print(img.shape)

            digit_list.append(img)
            label_list.append([int(number)])

    return digit_list, label_list


def main():
    digit_path = "data/"
    digit_list, label_list = get_digit_data(digit_path)

    digit_list = np.array(digit_list, dtype=np.float32)
    digit_list = digit_list.reshape(-1, DIGIT_HEIGHT * DIGIT_WIDTH)

    label_list = np.array(label_list)
    label_list = label_list.reshape(-1, 1)

    #  https://docs.opencv.org/3.4/d1/d2d/classcv_1_1ml_1_1SVM.html
    svm_model = cv2.ml.SVM_create()
    svm_model.setType(cv2.ml.SVM_C_SVC)  # C-Support Vector Classification
    # Histogram intersection kernel. A fast kernel.
    svm_model.setKernel(cv2.ml.SVM_INTER)
    # Termination criteria of the iterative SVM training procedure
    svm_model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    svm_model.train(digit_list, cv2.ml.ROW_SAMPLE, label_list)

    svm_model.save("svm.xml")


if __name__ == "__main__":
    main()
