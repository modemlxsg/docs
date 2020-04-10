import numpy as np
import torch
from shapely.geometry import Polygon
import shapely
import cv2
from FOTS.NMS import LA_NMS, MASK_NMS

def restore_polys_new(valid_pos, valid_geo, scale=4):
    polys = []
    index = []
    valid_pos *= scale  # Nx2
    dis = valid_geo[:4, :] # 4 x N
    angle = valid_geo[4,:]  # N,

    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        y_max = y + dis[0, i]
        x_max = x + dis[1, i]
        y_min = y - dis[2, i]
        x_min = x - dis[3, i]
        rect = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

        poly = Polygon(rect)
        poly = shapely.affinity.rotate(poly, -angle[i], origin=(x, y))
        poly = np.array(poly.exterior.coords)[:4]

        polys.append(poly)

    return np.array(polys)   

def get_boxes_new(score_map, geo_map, score_thresh=0.5, nms_thresh=0.2, scale=1):
    score_map = score_map[0]
    xy_text = np.argwhere(score_map > score_thresh)  # N*2
    if xy_text.size == 0:
        return None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:,::-1].copy()  # n x 2, [x, y]
    valid_geo = geo_map[:, xy_text[:, 0], xy_text[:, 1]]

    restore_polys = restore_polys_new(valid_pos, valid_geo, scale=scale)

    boxes = np.zeros((restore_polys.shape[0], 9), dtype=np.float32)
    boxes[:,:8] = restore_polys.reshape((-1,8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    boxes = LA_NMS.lanms(boxes.astype('float32'), nms_thresh)
    # boxes = MASK_NMS.mask_nms(boxes.astype('float32'), score_map.astype(np.uint8),nms_thresh)
    return boxes

def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def is_valid_poly(res, score_shape, scale):
    '''check if the poly in image scope
    Input:
        res        : restored poly in original image
        score_shape: score map shape
        scale      : feature map -> image
    Output:
        True if valid
    '''
    cnt = 0
    for i in range(res.shape[1]):
        if res[0,i] < 0 or res[0,i] >= score_shape[1] * scale or \
           res[1,i] < 0 or res[1,i] >= score_shape[0] * scale:
            cnt += 1
    return True if cnt <= 1 else False

def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
    '''restore polys from feature maps in given positions
    Input:
        valid_pos  : potential text positions <numpy.ndarray, (n,2)>
        valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
        score_shape: shape of score map
        scale      : image / feature map
    Output:
        restored polys <numpy.ndarray, (n,8)>, index
    '''

    polys = []
    index = []
    valid_pos *= scale  # Nx2
    d = valid_geo[:4, :] # 4 x N
    angle = valid_geo[4,:]  # N,
    
    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        y_max = y + d[0, i]
        x_max = x + d[1, i]
        y_min = y - d[2, i]
        x_min = x - d[3, i]
        rotate_mat = get_rotate_mat(-angle[i])

        # move to (0,0) rotate then move back
        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordidates = np.concatenate((temp_x, temp_y), axis=0)
        res = np.dot(rotate_mat, coordidates)
        res[0,:] += x
        res[1,:] += y

        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1], res[0, 2], res[1, 2], res[0, 3], res[1, 3]])

    return np.array(polys), index        

def get_boxes(score_map, geo_map, score_thresh=0.5, nms_thresh=0.2, scale=4):
    '''get boxes from feature map
    Input:
        score_map       : score map from model <numpy.ndarray, (1,row,col)>
        geo_map         : geo map from model <numpy.ndarray, (5,row,col)>
        score_thresh    : threshold to segment score map
        nms_thresh      : threshold in nms
    Output:
        boxes           : final polys <numpy.ndarray, (n,9)>
    '''

    score_map = score_map[0]
    xy_text = np.argwhere(score_map > score_thresh)  # N*2
    if xy_text.size == 0:
        return None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:,::-1].copy()  # n x 2, [x, y]
    valid_geo = geo_map[:, xy_text[:, 0], xy_text[:, 1]]
    polys_restored, index = restore_polys(valid_pos, valid_geo, score_map.shape, scale=scale)
    if polys_restored.size == 0:
        return None

    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:,:8] = polys_restored
    boxes[:, 8] = score_map[xy_text[index, 0], xy_text[index, 1]]
    boxes = lanms(boxes.astype('float32'), nms_thresh)
    return boxes
    

