import cv2
import imutils
import numpy as np
from math import *
import random
import datetime
import base64
import io
import sys
import math

glocal = True


def show_image(windowname, image, wait=False, local=glocal):
    if local:
        cv2.imshow(windowname, image)
        if wait:
            cv2.waitKey()
    return


def check_rectangle(c):
    approx_epsilon = 0.02
    approx = cv2.approxPolyDP(c, approx_epsilon * cv2.arcLength(c, True), True)
    num_of_points = len(approx)
    check_points = 4 <= num_of_points <= 5  # Dung la 19
    return check_points


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def detect4Conners(image):
    ret_code = 0

    # Sharpen image

    # Threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Find black box
    fine_contours = []
    # loop over our contours
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)

        if (10 < w < 40) and (10 < h < 40) and (w / h >= 1) and check_rectangle(c):  # and check_rectangle(c):
            if cv2.contourArea(c) / w * h > 0.8:
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                fine_contours.append(c)

    print("Number of Mark  Contours found = " + str(len(fine_contours)))

    # Get bounding rect for all contour
    fine_contours = np.concatenate(fine_contours)

    # Determine and draw bounding rectangle
    rect = cv2.minAreaRect(fine_contours)
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(image, [box], 0, (0, 0, 255), 3)

    show_image('a', image)
    show_image('b', thresh, wait=True)

    print("Rect=", rect)
    if (rect[1][0] < rect[1][1]):
        ra = 90 - rect[2]
    else:
        ra = -rect[2]
    print("Ra=", ra)
    if (ra != 0):
        # rotate
        image = imutils.rotate(image, angle=90 - ra)
        thresh = imutils.rotate(thresh, angle=90 - ra)
    # image,image1 = crop_rect(image,rect)
    # image = crop_minAreaRect(image,rect)
    # thresh = crop_minAreaRect(thresh,rect)
    # # except:
    # #     ret_code = -1
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 1)
    # # show_image('a1', image1)
    show_image('a', image)
    show_image('b', thresh, wait=True)
    return ret_code, image, thresh


def detectStudentID_new(image, thresh):
    ret_code = 0
    studentID = ""
    # try:
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Detect Student ID Rect
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if (250 < w < 350) and (350 < h < 450) and (w / h < 1):# and check_rectangle(c):
            thresh = thresh[y:y + h, x:x + w]
            image = image[y:y + h, x:x + w]
            print("Crop student id")
            break


    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 1)
    # show_image('b', thresh, wait=True)

    # mask = np.zeros_like(thresh)
    # # Fix horizontal and vertical lines
    # minLineLength = 100
    # lines = cv2.HoughLinesP(image=thresh, rho=1, theta=np.pi, threshold=50, lines=np.array([]),
    #                         minLineLength=minLineLength, maxLineGap=5)
    #
    # a, b, c = lines.shape
    # for i in range(a):
    #     cv2.line(mask, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), 255, 1,
    #              cv2.LINE_AA)
    #
    # lines = cv2.HoughLinesP(image=thresh, rho=1, theta=np.pi / 2, threshold=50, lines=np.array([]),
    #                         minLineLength=minLineLength, maxLineGap=5)
    #
    # a, b, c = lines.shape
    # for i in range(a):
    #     cv2.line(mask, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), 255, 1,
    #              cv2.LINE_AA)

    # Detect by morpth
    cols = thresh.shape[1]
    horizontal_size = cols // 3
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.erode(thresh, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    show_image('c', horizontal, wait=True)

    # Detect by morpth
    rows = thresh.shape[0]
    verticalsize = rows // 10
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(thresh, verticalStructure, iterations=2)
    vertical = cv2.dilate(vertical, verticalStructure)
    # Show extracted vertical lines
    # vertical = cv2.erode(vertical, (3, 3),iterations=50)
    show_image('c', vertical, wait=True)

    mask = np.bitwise_or(vertical, horizontal)

    show_image('c', mask, wait=True)

    # cv2.imshow("horizontal", mask)
    # cv2.waitKey()

    show_image('c', mask, wait=True)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (31,31))
    # mask = cv2.dilate(mask,(11,1),iterations=3)
    # mask = cv2.blur(mask,(3,3))
    for row in range(mask.shape[0]):
        white = np.count_nonzero(mask[row, :])
        if (white / mask.shape[1] > 0.3):
            # Draw line
            mask[row, :] = 255

    for col in range(mask.shape[1]):
        white = np.count_nonzero(mask[:, col])
        if (white / mask.shape[0] > 0.3):
            # Draw line
            mask[:, col] = 255

    show_image('c', mask, wait=True)
    # cv2.waitKey()

    # thresh = cv2.erode(thresh, kernel=(29, 29))
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    fine_contours = []
    # loop over our contours
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)

        if (30 < w < 90) and (20 < h < 90):
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 1)
            fine_contours.append(c)

    show_image('a', image)
    show_image('b', thresh, wait=True)

    print("So contour total studentID =", len(fine_contours))
    show_image('a', image, wait=True)
    if len(fine_contours) != 60:
        ret_code = -1
        return ret_code, studentID

    fine_contours = sort_contours(fine_contours, method="left-to-right")[0]

    for i in np.arange(0, len(fine_contours), 10):

        # color = (255,0,0)
        cnts = sort_contours(fine_contours[i:i + 10], method="top-to-bottom")[0]

        choice = (0, 0)
        for (j, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            # Tao mask de xem muc do to mau cua contour
            total = np.count_nonzero(thresh[y:y + h, x:x + w])

            # Lap de chon contour to mau dam nhat
            if total > choice[0]:
                # print("Chon")
                choice = (total, j)

            # Lay dap an cua cau hien tai
        studentID += str(choice[1])
        # cv2.drawContours(image, [cnts[choice[1]]], -1, color, 3)

    print("Student ID =", studentID)

    # except:
    #     ret_code = -1

    if len(studentID) != 6:
        ret_code = -1

    return ret_code, studentID


def detectExamID_new(image, thresh):
    ret_code = 0
    ExamID = ""
    # try:
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if (100 < w < 300) and (10 < h < 500) and (w / h < 1) and check_rectangle(c):
            thresh = thresh[y - 1:y + h + 1, x - 1:x + w + 1]
            image = image[y - 1:y + h + 1, x - 1:x + w + 1]
            break
    show_image("a", image)
    show_image("b", thresh, wait=True)

    # Detect by morpth
    cols = thresh.shape[1]
    horizontal_size = cols // 3
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.erode(thresh, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    # Show extracted horizontal lines

    # Detect by morpth
    rows = thresh.shape[0]
    verticalsize = rows // 10
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(thresh, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    # Show extracted vertical lines

    mask = np.bitwise_or(vertical, horizontal)
    for row in range(mask.shape[0]):
        white = np.count_nonzero(mask[row, :])
        if (white / mask.shape[1] > 0.3):
            # Draw line
            mask[row, :] = 255

    for col in range(mask.shape[1]):
        white = np.count_nonzero(mask[:, col])
        if (white / mask.shape[0] > 0.3):
            # Draw line
            mask[:, col] = 255

    show_image("c", mask, wait=True)

    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    fine_contours = []
    # loop over our contours
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)

        if (10 < w < 60) and (10 < h < 80) and check_rectangle(c):  # and (w / h >= 1):
            # print(x, y, w, h)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # cv2.imshow('a', image)
            # cv2.waitKey()
            fine_contours.append(c)

    print("So contour total ExamID =", len(fine_contours))
    # cv2.drawContours(image, fine_contours, -1, (0, 0, 255), 3)
    # cv2.imshow('a', image)
    # cv2.imshow('b', thresh)
    # cv2.waitKey()
    # sys.exit()

    if len(fine_contours) != 30:
        ret_code = -1
        return ret_code, ExamID

    fine_contours = sort_contours(fine_contours, method="left-to-right")[0]
    # print("So contour total=",len(fine_contours))

    for i in np.arange(0, len(fine_contours), 10):

        # color = (255,0,0)
        cnts = sort_contours(fine_contours[i:i + 10], method="top-to-bottom")[0]

        choice = (0, 0)
        for (j, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            # Tao mask de xem muc do to mau cua contour
            total = np.count_nonzero(thresh[y:y + h, x:x + w])

            # Lap de chon contour to mau dam nhat
            if total > choice[0]:
                # print("Chon")
                choice = (total, j)

            # Lay dap an cua cau hien tai
        ExamID += str(choice[1])
        # cv2.drawContours(image, [cnts[choice[1]]], -1, color, 3)

    print("ExamID=", ExamID)
    # cv2.imshow('a', image)
    # cv2.imshow('b', thresh)
    # cv2.waitKey()
    # except:
    #     ret_code = -1

    return ret_code, ExamID


def detectResult(image, thresh):
    ret_code = 0
    cellResult = ""

    show_image('a', image)
    show_image('b', thresh, wait=True)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    gray = cv2.filter2D(image, -1, kernel)

    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # thresh_new = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 50)
    _, thresh_new = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    show_image('c', thresh_new, wait=True)
    thresh_new = cv2.dilate(thresh_new, (3, 3), iterations=1)

    for row in range(thresh_new.shape[0]):
        b2w = 0
        w2b = 0
        cpoint = thresh_new[row, 0]
        for col in range(1, thresh_new.shape[1]):
            if thresh_new[row, col] != cpoint:
                if cpoint == 0:
                    b2w += 1
                else:
                    w2b += 1
                cpoint = thresh_new[row, col]
                if (b2w >= 4) or (w2b >= 4):
                    break
        # print("Row {} - {} b2w - {} w2b".format(row,b2w, w2b))
        if b2w < 4 and w2b < 4:
            thresh_new[row, :] = 0

    for col in range(thresh_new.shape[1]):
        b2w = 0
        w2b = 0
        cpoint = thresh_new[0, col]
        for row in range(1, thresh_new.shape[0]):
            if thresh_new[row, col] != cpoint:
                if cpoint == 0:
                    b2w += 1
                else:
                    w2b += 1
                cpoint = thresh_new[row, col]
                if (b2w >= 4) or (w2b >= 4):
                    break
        # print("Col {} - {} b2w - {} w2b".format(row, b2w, w2b))

        if b2w < 4 and w2b < 4:
            thresh_new[:, col] = 0

    show_image('c', thresh_new, wait=True)
    # Draw 4 line
    row_height = thresh_new.shape[0]//5
    print("Row height =", row_height)
    for row in  range(1, thresh_new.shape[0]//row_height):
        thresh_new[row*row_height+5,:] = 0

    show_image('c', thresh_new, wait=True)

    contours = cv2.findContours(thresh_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    fine_contours = []
    # loop over our contours
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if (20 < w < 65) and (20 < h < 85):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            fine_contours.append(c)

    show_image('a', image, wait=True)

    if len(fine_contours) < 20:
        print("1Len cont=", len(fine_contours))
        print("1Not enough contours!")
        ret_code = -1
        return ret_code, cellResult

    fine_contours = sorted(fine_contours, key=cv2.contourArea, reverse=True)
    fine_contours = fine_contours[:20]

    # loop 4 contours
    fine_contours = sort_contours(fine_contours, method="top-to-bottom")[0]

    for i in np.arange(0, len(fine_contours), 4):

        cnts = sort_contours(fine_contours[i:i + 4], method="left-to-right")[0]

        choice = (-1, -1)
        cnt_size = []

        for (j, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)

            extracted = np.zeros_like(thresh)
            cv2.drawContours(extracted, [c], -1, 255, cv2.FILLED)
            masked_thresh = cv2.bitwise_and(thresh_new, extracted)

            total = np.count_nonzero(masked_thresh) / (w * h)
            if total > choice[0]:
                choice = (total, j)

            cnt_size.append(total)

        print("Row ", i // 4)
        # avg_weights = sum(cnt_size)/len(cnt_size)
        # print("AVG Weights=",avg_weights)

        # for idx in range(len(cnt_size)):
        #         if cnt_size[idx]>choice[0]:
        #             choice = (cnt_size[idx],idx)

        count = 0
        for idx in range(len(cnt_size)):
            if (cnt_size[idx] >= 0.8 * choice[0]) and (cnt_size[idx] >= 0.36):
                count += 1

        print("Choice=", choice)
        print("Count=", count)

        if count == 1:
            cellResult += str(chr(65 + choice[1]))
        elif count > 1:
            cellResult += "*"
        else:
            cellResult += "-"

    return ret_code, cellResult


def detectResultSheet_new(image, thresh):
    start = datetime.datetime.now()
    ret_code = 0
    result = ""
    # Resize image
    image = image[image.shape[0] * 2 // 5:, :]
    thresh = thresh[thresh.shape[0] * 2 // 5:, :]

    # try:
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # loop over our contours
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        # print(x, y, w, h)
        if (w / h <= 1.5) and (800 < w < 1200):
            thresh = thresh[y:y + h, x:x + w]
            image = image[y:y + h, x:x + w]
            print("Crop")
            break

    show_image('a', image, wait=True)

    # Detect by h line
    cols = thresh.shape[1]
    horizontal_size = cols // 4
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(thresh, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure, iterations=2)

    # Detect by v line
    rows = thresh.shape[0]
    verticalsize = rows // 6
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(thresh, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure, iterations=2)

    mask = np.bitwise_or(vertical, horizontal)

    for row in range(mask.shape[0]):
        white = np.count_nonzero(mask[row, :])
        if white / mask.shape[1] > 0.5:
            mask[row, :] = 255

    for col in range(mask.shape[1]):
        white = np.count_nonzero(mask[:, col])
        if white / mask.shape[0] > 0.5:
            mask[:, col] = 255

    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    fine_contours = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)

        if (100 < w < 700) and (100 < h < 700):
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
            fine_contours.append(c)

    show_image('a', image, wait=True)

    print("So contour total ResultSheet =", len(fine_contours))

    if len(fine_contours) != 24:
        ret_code = -1
        return ret_code, result

    fine_contours = sort_contours(fine_contours, method="left-to-right")[0]

    currentCell = 0

    for i in np.arange(0, len(fine_contours), 6):

        # color = (255,0,0)
        cnts = sort_contours(fine_contours[i:i + 6], method="top-to-bottom")[0]

        for (j, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)

            currentCell += 1
            print("---------Current cell=", currentCell, (x, y, w, h))
            # show_image('a',image[y-3:y + h+3, x-3:x + w+3])
            padding =0
            ret_code, cellResult = detectResult(image[y-padding:y + h+padding, x:x + w], thresh[y-padding:y + h+padding, x:x + w])

            if ret_code != 0:
                ret_code = -1
                return ret_code, result

            result += cellResult
            print(cellResult)

    print("Ket qua sheet = ", result)
    print("Thoi gian xu ly Sheet = ", datetime.datetime.now() - start)
    return ret_code, result


def process(image):
    start = datetime.datetime.now()
    final_info = ""
    # image = cv2.imread('data/P1-0005a.jpg')
    # Step 1. Detect 4 corners and rotate
    ret_code, image, thresh = detect4Conners(image)
    if ret_code != 0:
        retdata = {"ret_code": -1, "info": "Can not detect marked points."}
        return (retdata)

    # Step 2. Detect Student ID
    print("Run detectStudentID...")
    ret_code, studentID = detectStudentID_new(image, thresh)
    print("Return from detectStudentID = ", ret_code, studentID)
    if ret_code != 0:
        print("Exit")
        retdata = {"ret_code": -1, "info": "Can not detect studentID."}
        return (retdata)
    else:
        print("Go")
        final_info += studentID

    # Step 3. Detect Exam ID
    print("Run detectExamID...")
    ret_code, ExamID = detectExamID_new(image, thresh)
    print("Return from detectExamID = ", ret_code, ExamID)
    if ret_code != 0:
        retdata = {"ret_code": -1, "info": "Can not detect ExamID."}
        return (retdata)
    else:
        final_info += ExamID

    print("Run detectResultSheet...")
    # Step 4. Detect Result
    ret_code, resultSheet = detectResultSheet_new(image, thresh)
    if ret_code != 0:
        retdata = {"ret_code": -1, "info": "Can not detect resultSheet."}
        return (retdata)
    else:
        final_info += resultSheet

    print(final_info)

    # result, point = calculate_points(final_info)
    print("Thoi gian xu ly = ", datetime.datetime.now() - start)
    # image = detectResultSheet(image)
    retdata = {
        "ret_code": 0,
        "info": final_info
    }

    # except:
    #     print("except")
    #     retdata = {
    #         "ret_code": -1,
    #         "info": "Undefined exception."
    #     }

    return (retdata)


def mark():
    global glocal
    # Read image
    # try:
    path = "data/bai thi 120 cau"
    one = True
    if one:
        glocal = True
        image = cv2.imread('data/bai thi 120 cau/P9-0001.jpg')
        result = process(image)
        print(result)
    else:
        import os
        glocal = False
        #path = "data/Bai 120 cau  - new"
        file1 = open("result.txt", "w")
        for filename in os.listdir(path):
            if filename.endswith(".jpg") and (not filename.startswith(".")):
                # print(os.path.join(directory, filename))
                print("*******************************" + path + "/" + filename)
                image = cv2.imread(path + "/" + filename)  # 'data/Bai 120 cau  - new/P2 (8).jpg')
                result = process(image)
                print(result)
                file1.write(filename + " - " + str(result.get("info")) + "\n")
        file1.close()


print(mark())
