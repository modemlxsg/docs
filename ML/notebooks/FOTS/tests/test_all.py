# import sys
# import os
# sys.path.append(os.getcwd())
# sys.path.append("H:/docs/ML/notebooks/FOTS")

# import pytest
# from FOTS.model import *
# import numpy as np
# from FOTS.utils import get_boxes
# import time
# import cv2

# x = torch.randn((1, 3, 512, 512), dtype=torch.float32)
# im = cv2.imread("H:/docs/ML/notebooks/FOTS/demo/imgs/img_310.jpg")
# im = im[:,:,::-1]
# im = im.transpose((2, 0, 1))
# print(im.shape,type(im))

# net1 = SharedConv()
# net2 = Detector()
# o = net1(x)
# score_map, geo_map = net2(o)

# score_map = score_map[0].detach().numpy()
# geo_map = geo_map[0].detach().numpy()

# print(score_map.shape, geo_map.shape)

# boxes = get_boxes(score_map, geo_map)
# print(boxes)

# img = np.zeros((512, 512))
# boxes = np.intp(boxes)
# box = boxes[:,:8].reshape(4, 2)
# print(box)

# cv2.polylines(img, [box], True, (255, 0, 0))
# cv2.imshow('1', img)
# cv2.waitKey()
# cv2.destroyAllWindows()



