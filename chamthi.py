import cv2
import imutils
import numpy as np
from math import *
import random
import datetime

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    maxHeight, maxWidth = image.shape[:2]

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 2],
        [maxWidth - 1, 2],
        [maxWidth - 1, maxHeight],
        [0, maxHeight]], dtype="float32")
    #print(image.shape)

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight),borderValue=0)
    # return the warped image
    return warped

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
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]+90
    #print(angle)
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    img_rot = imutils.rotate(img,angle)

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]

    return img_rot

def detect4Conners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 1)
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Find black box
    number = 0
    fine_contours = []
    # loop over our contours
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)

        # Check for rectangle
        approx_epsilon = 0.02
        approx = cv2.approxPolyDP(c, approx_epsilon * cv2.arcLength(c, True), True)
        num_of_points = len(approx)
        check_points = 4 <= num_of_points <= 5  # Dung la 19

        if (20 < w < 30) and (10 < h < 25) and (w/h>1) and check_points:
            if cv2.contourArea(c)/w*h>0.8:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                number += 1
                fine_contours.append(c)

    #print("Number of Contours found = " + str(number))

    # Get bounding rect for all contour
    fine_contours = np.concatenate(fine_contours)

    # Determine and draw bounding rectangle
    rect = cv2.minAreaRect(fine_contours)
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(image, [box], 0, (0, 0, 255), 3)

    # cv2.imshow('a', image)
    # cv2.imshow('b', thresh)
    # cv2.waitKey()

    image = crop_minAreaRect(image,rect)
    thresh = crop_minAreaRect(thresh,rect)


    return image, thresh

def detectStudentID(image, thresh):

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)



    # Detect Student ID Rect
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        #print(x, y, w, h)

        approx_epsilon = 0.02
        approx = cv2.approxPolyDP(c, approx_epsilon * cv2.arcLength(c, True), True)
        num_of_points = len(approx)
        check_points = 4 <= num_of_points <= 5  # Dung la 19

        if (300 < w < 400) and (400 < h < 800) and (w/h<1) and check_points:
                thresh = thresh[y:y+h,x:x+w]
                image = image[y:y+h,x:x+w]
                break


    thresh = cv2.erode(thresh, kernel=(29, 29))
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, (19, 19))
    # cv2.imshow('a', image)
    # cv2.imshow('b', thresh)
    # cv2.waitKey()

    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    fine_contours = []
    # loop over our contours
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        #print(x, y, w, h)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # approximate the contour
        approx_epsilon = 0.02
        approx = cv2.approxPolyDP(c, approx_epsilon * cv2.arcLength(c, True), True)
        num_of_points = len(approx)
        check_points = 4 <= num_of_points <= 4 # Dung la 19

        if (40 < w < 90) and (20 < h < 90):# and check_points:
                cv2.rectangle(image, (x, y), (x + w, y + h), (random.randint(0,255), random.randint(0,255), random.randint(0,255)), 2)
                fine_contours.append(c)

    # cv2.imshow('a', image)
    # cv2.imshow('b', thresh)
    # cv2.waitKey()
    fine_contours = sort_contours(fine_contours,method="left-to-right")[0]
    print("So contour total=",len(fine_contours))


    # return

    studentID = ""
    for i in np.arange(0,len(fine_contours),10):

        color = (255,0,0)
        cnts = sort_contours(fine_contours[i:i+10],method="top-to-bottom")[0]

        choice = (0, 0)
        for (j, c) in enumerate(cnts):
            (x,y,w,h) = cv2.boundingRect(c)
            # Tao mask de xem muc do to mau cua contour
            total = np.count_nonzero(thresh[y:y+h,x:x+w])

            # Lap de chon contour to mau dam nhat
            if total > choice[0]:
                #print("Chon")
                choice = (total, j)

            # Lay dap an cua cau hien tai
        studentID += str(choice[1])
        cv2.drawContours(image, [cnts[choice[1]]], -1, color, 3)

    print("Student ID =",studentID)
    #cv2.imshow('a', image)
    #cv2.imshow('b', thresh)
    #cv2.waitKey()

    return studentID

def detectExamID(image, thresh):

    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)

        # approximate the contour
        approx_epsilon = 0.02
        approx = cv2.approxPolyDP(c, approx_epsilon * cv2.arcLength(c, True), True)
        num_of_points = len(approx)
        check_points = 4 <= num_of_points <= 5  # Dung la 19

        if (100 < w < 300) and (10 < h < 800) and (w/h<1) and check_points:
                thresh = thresh[y:y+h,x:x+w]
                image = image[y:y + h, x:x + w]
                break

    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    fine_contours = []
    # loop over our contours
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)

        # approximate the contour
        approx_epsilon = 0.02
        approx = cv2.approxPolyDP(c, approx_epsilon * cv2.arcLength(c, True), True)
        num_of_points = len(approx)
        check_points = 4 <= num_of_points <= 5  # Dung la 19

        if (40 < w < 80) and (20 < h < 60) and (w / h >= 1):# and check_points:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                fine_contours.append(c)

    # cv2.imshow('a', image)
    # cv2.imshow('b', thresh)
    # cv2.waitKey()
    # return
    fine_contours = sort_contours(fine_contours,method="left-to-right")[0]
    #print("So contour total=",len(fine_contours))
    ExamID = ""
    for i in np.arange(0,len(fine_contours),10):

        color = (255,0,0)
        cnts = sort_contours(fine_contours[i:i+10],method="top-to-bottom")[0]

        choice = (0, 0)
        for (j, c) in enumerate(cnts):
            (x,y,w,h) = cv2.boundingRect(c)
            # Tao mask de xem muc do to mau cua contour
            total = np.count_nonzero(thresh[y:y+h,x:x+w])

            # Lap de chon contour to mau dam nhat
            if total > choice[0]:
                #print("Chon")
                choice = (total, j)

            # Lay dap an cua cau hien tai
        ExamID += str(choice[1])
        cv2.drawContours(image, [cnts[choice[1]]], -1, color, 3)

    print("ExamID=",ExamID)
    # cv2.imshow('a', image)
    # cv2.imshow('b', thresh)
    # cv2.waitKey()

    return ExamID

def detectResult(image, thresh):
    thresh = thresh[3:thresh.shape[0]-3,5:thresh.shape[1]-5]
    image = image[3:image.shape[0] - 3, 5:image.shape[1] - 5]
    #thresh = cv2.erode(thresh, kernel=(5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, (3,3))

    #thresh = cv2.blur(thresh, ksize=(3,3))

    #thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    number = 0
    fine_contours = []
    #print("--------------------")
    # loop over our contours
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        #if (w>10) and (h>10):
            #print(x, y, w, h,w/h)
        if (25 < w < 60) and (25 < h < 60):# and (0.5< w / h < 1.5):
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                number += 1
                fine_contours.append(c)

    fine_contours = sorted(fine_contours, key=cv2.contourArea, reverse=True)
    fine_contours = fine_contours[:20]

    fine_contours = sort_contours(fine_contours, method="top-to-bottom")[0]
    #print("So contour total=", len(fine_contours))

    cellResult = ""
    for i in np.arange(0, len(fine_contours), 4):

        color = (255, 0, 0)
        cnts = sort_contours(fine_contours[i:i + 4], method="left-to-right")[0]

        choice = (-1, -1)
        cnt_size = []
        count =0

        for (j, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)

            total = np.count_nonzero(thresh[y:y + h, x:x + w])
            cnt_size.append(total)

        avg_weights = sum(cnt_size)/len(cnt_size)
        print(avg_weights)


        for idx in range(len(cnt_size)):
            if cnt_size[idx]>avg_weights:
                choice = (cnt_size[idx],idx)
                count+=1

        # if (total > choice[0])and(total/(w*h)>0.3):
        #     #print("Chon")
        #     choice = (total, j)


        if count==1:
            cellResult+= str(chr(65+choice[1]))
            cv2.drawContours(image, cnts[choice[1]], -1, color, 2)
        else:
            if avg_weights>480:
                cellResult+="*"
            else:
                cellResult+="-"

    # cv2.imshow('a', image)
    # cv2.imshow('b', thresh)
    # cv2.waitKey()

    return cellResult

def detectResultSheet(image, thresh):
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)


    number = 0
    fine_contours = []
    # loop over our contours
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)

        approx_epsilon = 0.02
        approx = cv2.approxPolyDP(c, approx_epsilon * cv2.arcLength(c, True), True)
        num_of_points = len(approx)
        check_points = 4 <= num_of_points <= 5  # Dung la 19

        if (100 < w < 500) and (100 < h < 500) and (w/h<=1.5) and (y>image.shape[1]*1/3): #and check_points:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 3)
                number += 1
                fine_contours.append(c)

    # print("So contour total=", len(fine_contours))
    # cv2.imshow('aaa', image)
    # cv2.imshow('bbb', thresh)
    # cv2.waitKey()

    fine_contours = sort_contours(fine_contours, method="left-to-right")[0]


    currentCell = 0
    result=""
    for i in np.arange(0,len(fine_contours),6):

        color = (255,0,0)
        cnts = sort_contours(fine_contours[i:i+6],method="top-to-bottom")[0]

        for (j, c) in enumerate(cnts):
            (x,y,w,h) = cv2.boundingRect(c)
            currentCell +=1
            print("---------Current cell=",currentCell)
            cellResult = detectResult(image[y:y+h,x:x+w],thresh[y:y+h,x:x+w])
            result +=cellResult
            print(cellResult)

        cv2.drawContours(image, cnts, -1, color, 3)



    # cv2.imshow('a', image)
    # cv2.imshow('b', thresh)
    # cv2.waitKey()

    print(result)
    return result

def calculate_points(final_info):
    exam_id = final_info[6:9]
    print("Checking, examid=",exam_id)
    result = final_info[9:]
    print("Checking, result=", result)
    right = right_answer.get(exam_id)
    print("Right answer=", right)
    point = 0
    for i in range(len(result)):
        if result[i] == right[i]:
            point+=1

    print("Point = {}/{}".format(point,len(result)))
    return



right_answer = {
    "132":"BACBBDDCDCDBBACBCDDAAAABBCADADCCABCCDBDDCDADACAABADAABACCDCBADBDBCBDCCBDBCCBBADDCADCDCDCCBABAABAACDBBDDADCADBCBCDCCDDABD",
    "209":"BBDBCDCDDBDAACCCBDAAADBBCDBCDCAAABDDBDACCAACDADBDCCABABADCAADDACDCCCBBCBBDBBADBDDBDADBDABADCADDBCBBCDCDCCCDACBABACDDABCA",
    "357":"DABDDDBDBCACCBAAABDCCAAAAACACBCCDDBBDDBAABADDCBBAAABADDBADDCDBACDCCABCCDBBBCDCACADDACBAABCCBCDBCBBCDADAADDAAABDCCCDBAABB",
    "485":"BDDABBDACACDCACBCDBBADCCDCBDBACBDDBABAACACDDBBBDCDBDADBADBCAAADCBDCAABDBBCCCAACCADCDBDBCACCDAAACBDDCBBDDDBBDACCCABBAACAA"
}

start = datetime.datetime.now()

image = cv2.imread('data/P1-0005a.jpg')
# Step 1. Detect 4 corners and rotate
image, thresh = detect4Conners(image)

# cv2.imshow('a', image)
# cv2.waitKey()


# Step 2. Detect Student ID
final_info  = detectStudentID(image, thresh)

# Step 3. Detect Exam ID
final_info += detectExamID(image, thresh)

# Step 4. Detect Result
final_info += detectResultSheet(image, thresh)

print(final_info)

calculate_points(final_info)
print("Thoi gian xu ly = ",datetime.datetime.now()-start)
#image = detectResultSheet(image)


