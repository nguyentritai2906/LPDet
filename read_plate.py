import argparse

import cv2
import numpy as np
from lib_detection import detect_lp, im2single, load_model

parser = argparse.ArgumentParser()
parser.add_argument("input", nargs='?', metavar="INPUT", type=str)

CHAR_LIST = '0123456789ABCDEFGHKLMNPRSTUVXYZ'
DIGIT_WIDTH = 30
DIGIT_HEIGHT = 60
THRESHOLD_UPPER_BOUND = 3.5
THRESHOLD_LOWER_BOUND = 1.5


def sort_contours(cnts):
    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(
        zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts


def remove_unused_chars(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in CHAR_LIST:
            newString += lp[i]
    return newString


def get_min_dim(Ivehicle, Dmin=288, Dmax=608):
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    return min(side, Dmax)


def showInMovedWindow(winname, img, binary, roi, x, y):
    w = img.shape[1]
    bin_h, bin_w = binary.shape
    r = float(w / bin_w)
    h = int(bin_h * r)

    binary = cv2.resize(binary, (w, h), None)
    roi = cv2.resize(roi, (w, h), None)
    grey_3_channel = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    numpy_vertical = np.vstack((roi, grey_3_channel, img))
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, x, y)  # Move it to (x,y)
    while True:
        cv2.imshow(winname, numpy_vertical)
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break


def main(args):
    # Load Warped Planar Object Detection Network (WPOD-NET) model
    wpod_net_path = "wpod-net_update1.json"
    wpod_net = load_model(wpod_net_path)

    Ivehicle = cv2.imread(args.input)

    bound_dim = get_min_dim(Ivehicle)

    _, LpImg, lp_type = detect_lp(wpod_net,
                                  im2single(Ivehicle),
                                  bound_dim,
                                  lp_threshold=0.5)

    model_svm = cv2.ml.SVM_load('svm.xml')

    if (len(LpImg)):

        # Chuyen doi anh bien so
        LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
        roi = LpImg[0]

        gray = cv2.cvtColor(LpImg[0], cv2.COLOR_BGR2GRAY)

        # Ap dung threshold de phan tach so va nen
        binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]

        # Segment chars
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        cont, _ = cv2.findContours(thre_mor, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

        plate_info = ""

        for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h / w
            if THRESHOLD_LOWER_BOUND <= ratio <= THRESHOLD_UPPER_BOUND:
                if h / roi.shape[0] >= 0.6:  # Height of contour >= ROI height

                    # Draw rectangle around number
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Split & predict
                    curr_num = thre_mor[y:y + h, x:x + w]
                    curr_num = cv2.resize(curr_num,
                                          dsize=(DIGIT_WIDTH, DIGIT_HEIGHT))
                    _, curr_num = cv2.threshold(curr_num, 30, 255,
                                                cv2.THRESH_BINARY)
                    curr_num = np.array(curr_num, dtype=np.float32)
                    curr_num = curr_num.reshape(-1, DIGIT_WIDTH * DIGIT_HEIGHT)

                    result = model_svm.predict(curr_num)[1]
                    result = int(result[0, 0])

                    if result <= 9:  # Is Number
                        result = str(result)
                    else:  # ASCII convertion
                        result = chr(result)

                    plate_info += result

        # Draw found LP on Img
        cv2.putText(Ivehicle,
                    remove_unused_chars(plate_info), (50, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    3.0, (0, 0, 255),
                    lineType=cv2.LINE_AA,
                    thickness=4)

        print("License Plate: ", plate_info)
        showInMovedWindow("Output", Ivehicle, binary, roi, 30, 40)


if __name__ == "__main__":
    main(parser.parse_args())
