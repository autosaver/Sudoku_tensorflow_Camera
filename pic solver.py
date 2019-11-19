import cv2
import argparse
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import operator
import copy
import os
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from skimage.segmentation import clear_border
from keras.models import load_model
ap = argparse.ArgumentParser()
args = vars(ap.parse_args())

#Show Image in cv2 window
def show_image(img,title):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 450,450)
    cv2.imshow(title, img)
    cv2.waitKey(50)
    cv2.destroyAllWindows()

#Image filter processing
def pre_process_image(img, skip_dilate=False):
    proc = cv2.GaussianBlur(img.copy(),(11,11),0)   #(w,h) odd and +ive
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    proc = cv2.bitwise_not(proc, proc)
    if not skip_dilate:
      kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
      proc = cv2.dilate(proc, kernel)
    return proc

#Find image corners
def findCorners(img):
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

#Function used to specify point
def display_points(in_img, points, radius=5, colour=(0, 0, 255)):
    img = in_img.copy()
    if len(colour) == 3:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for point in points:
        cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)
    #show_image(img,"display_points")
    return img


def distance_between(p1, p2):
	a = p2[0] - p1[0]
	b = p2[1] - p1[1]
	return np.sqrt((a ** 2) + (b ** 2))

def display_rects(in_img, rects, colour=255):
    img = in_img.copy()
    for rect in rects:
        cv2.rectangle(img, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), colour)
    #show_image(img,"display_rects")
    return img

def crop_and_warp(img, crop_rect):
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    side = max([
		distance_between(bottom_right, top_right),
		distance_between(top_left, bottom_left),
		distance_between(bottom_right, bottom_left),
		distance_between(top_left, top_right)
	])
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (int(side), int(side)))

def infer_grid(img):
	squares = []
	side = img.shape[:1]
	side = side[0] / 9
	for i in range(9):
		for j in range(9):
			p1 = (i * side, j * side)
			p2 = ((i + 1) * side, (j + 1) * side)
			squares.append((p1, p2))
	return squares

def getEveryDigits(img,squares):
    labels = []
    centers = []
    img2=img.copy()
    #show_image(img2,"TEST")
    height, width = img.shape[:2]
    img2 = Image.fromarray(img2)
    for i in range(81):
        x1=(int)(squares[i][0][0])
        x2=(int)(squares[i][1][0])
        y1=(int)(squares[i][0][1])
        y2=(int)(squares[i][1][1])
        window=img[x1:x2, y1:y2]
        #print(window.shape)
        #show_image(window,"Window")
        digit = cv2.resize(window,(28,28))
        #show_image(digit,"TEST")
        digit = clear_border(digit)
        #show_image(digit,"TEST")
        
        #print(digit.shape)
        #show_image(digit,"TEST")
        numPixels = cv2.countNonZero(digit)
        if numPixels<minreqpixels:
            label=0
        else:   
            label2 = model.predict_classes([digit.reshape(1,28,28,1)])
            label=label2[0]
        
        labels.append(label)
    return matrix_convert(labels)

def matrix_convert(label):
  a=0
  matrix=[]
  for i in range(0,9):
        matrix.append(label[a:a+9])
        a=a+9
  return matrix

def checkGrid(grid):
  for row in range(0,9):
      for col in range(0,9):
        if grid[row][col]==0:
          return False
  return True
#----------Solving FUNCTIONS-----------------# 


def solve(bo):
    find = find_empty(bo)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1,10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i

            if solve(bo):
                return True

            bo[row][col] = 0

    return False


def valid(bo, num, pos):
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if bo[i][j] == num and (i,j) != pos:
                return False

    return True


def print_board(bo):
    for i in range(len(bo)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")

        for j in range(len(bo[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")

            if j == 8:
                print(bo[i][j])
            else:
                print(str(bo[i][j]) + " ", end="")


def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return (i, j)  # row, col

    return None
print("\n-----ϟϴÐṲḰṲ------\n")
minreqpixels=90
model = load_model('models/mnist_keras_cnn_model.h5')
#img = cv2.imread('image4.jpg', cv2.IMREAD_GRAYSCALE)
while 1:
        cam=cv2.VideoCapture('http://10.42.0.209:8080/video')
        _,img=cam.read()
        #img = cv2.imread(img)
        if img is None:
             continue
        original=img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed = pre_process_image(img)
        corners = findCorners(processed)
        cropped = crop_and_warp(processed, corners)
        squares = infer_grid(cropped)
        old= getEveryDigits(cropped,squares)
        #show_image(img,"Original Image")
        #show_image(processed,"Processed Image")
        display_points(processed, corners)
        board = np.copy(old)
        originalboard=np.copy(old)
        solve(board)
        if any(0 in x for x in board):
            print("Can't Scan Sudoku please hold still")
            continue
        print_board(originalboard)
        print("   ___________________  ")
        print("  \n Solved board:-> ")
        print_board(board)
        input("\n Press Enter to continue...")
        








        
