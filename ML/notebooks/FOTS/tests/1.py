import sys
import os
sys.path.append(os.getcwd())
sys.path.append("H:/docs/ML/notebooks/FOTS")

import cv2
import torch
from FOTS.model import *
from FOTS.utils import *

import imgaug as ia
import imgaug.augmenters as iaa

augs = iaa.Sequential([
    iaa.PadToSquare(),
    iaa.Resize((512,512))
])

im = cv2.imread("H:/docs/ML/notebooks/FOTS/demo/imgs/img_310.jpg")
print(im.shape)

im = np.zeros((1000, 1000, 32))


# 733,113,966,60,969,150,732,187,E-PAY
# 784,317,847,317,844,338,782,338,AXS

box1 = np.array([[733, 113], [966, 60], [969, 150], [732, 187]])
box2 = np.array([[784,317], [847,317], [844,338], [782,338,]])
#cv2.polylines(im, [box1,box2], True, (0, 0, 255))

rotate_box = cv2.minAreaRect(box1)
print(rotate_box)
rect = cv2.boxPoints(rotate_box)
print(rect)
rect_int = np.intp(rect)
#cv2.polylines(im, [rect_int], True, (255, 0, 0))

b_w, b_h = rotate_box[1][0], rotate_box[1][1]

spts = np.array([
    (box1[0][0], box1[0][1]),
    (box1[1][0], box1[1][1]),
    (box1[3][0], box1[3][1])
])
dpts = np.array([
    (0, 0),
    (b_w, 0),
    (0, b_h)
])


matrix = cv2.getAffineTransform(spts.astype(np.float32), dpts.astype(np.float32))
# matrix = cv2.getRotationMatrix2D(rotate_box[0], rotate_box[2], 1.0)
print(matrix)
print(im.shape)

im_affine = cv2.warpAffine(im, matrix, (im.shape[1],im.shape[0]))

b_w = np.round(b_w).astype(np.int32)
b_h = np.round(b_h).astype(np.int32)
print(b_w,b_h)
roi = im_affine[0:b_h, 0:b_w,:]

print(roi.shape)

# cv2.imshow("1", im_affine)
# cv2.imshow("2", roi)
# cv2.waitKey()
# cv2.destroyAllWindows()







