from shapely.geometry import Polygon
import numpy as np
import copy
import cv2


class LA_NMS:

    @staticmethod
    def intersection(g, p):
        #取g,p中的几何体信息组成多边形
        g = Polygon(g[:8].reshape((4, 2)))
        p = Polygon(p[:8].reshape((4, 2)))

        # 判断g,p是否为有效的多边形几何体
        if not g.is_valid or not p.is_valid:
            return 0

        # 取两个几何体的交集和并集
        inter = Polygon(g).intersection(Polygon(p)).area
        union = g.area + p.area - inter
        if union == 0:
            return 0
        else:
            return inter/union

    @staticmethod
    def weighted_merge(g, p):
        # 取g,p两个几何体的加权（权重根据对应的检测得分计算得到）
        g[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])

        #合并后的几何体的得分为两个几何体得分的总和
        g[8] = (g[8] + p[8])
        return g

    @staticmethod
    def standard_nms(S, thres):
        #标准NMS
        order = np.argsort(S[:, 8])[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            ovr = np.array([LA_NMS.intersection(S[i], S[t]) for t in order[1:]])
            inds = np.where(ovr <= thres)[0]
            order = order[inds+1]

        return S[keep]

    @staticmethod
    def lanms(polys, thres=0.8):
        '''
        locality aware nms of EAST
        :param polys: a N*9 numpy array. first 8 coordinates, then prob
        :return: boxes after nms
        '''
        S = []    #合并后的几何体集合
        p = None  #合并后的几何体
        for g in polys:
            if p is not None and LA_NMS.intersection(g, p) > thres:#若两个几何体的相交面积大于指定的阈值，则进行合并
                p = LA_NMS.weighted_merge(g, p)
            else:    #反之，则保留当前的几何体
                if p is not None:
                    S.append(p)
                p = g
        if p is not None:
            S.append(p)
        if len(S) == 0:
            return np.array([])
        return LA_NMS.standard_nms(np.array(S), thres)

class MASK_NMS:

    EPS=0.00001
    
    staticmethod
    def get_mask(box,mask):
        """根据box获取对应的掩膜"""
        tmp_mask = np.zeros(mask.shape, dtype="uint8")
        tmp = np.array(box.tolist(), dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(tmp_mask, [tmp], (255))
        
        tmp_mask = cv2.bitwise_and(tmp_mask, mask)

        return tmp_mask, cv2.countNonZero(tmp_mask)
        

    @staticmethod
    def comput_mmi(area_a,area_b,intersect):
        """
        计算MMI
        :param mask_a: 实例文本a的mask的面积
        :param mask_b: 实例文本b的mask的面积
        :param intersect: 实例文本a和实例文本b的相交面积
        :return:
        """
        if area_a==0 or area_b==0:
            area_a+=MASK_NMS.EPS
            area_b+=MASK_NMS.EPS
            print("the area of text is 0")
        return max(float(intersect)/area_a,float(intersect)/area_b)

    @staticmethod
    def mask_nms(dets, mask, thres=0.3):
        """
        mask nms 实现函数
        :param dets: 检测结果，是一个N*9的numpy,
        :param mask: 当前检测的mask
        :param thres: 检测的阈值
        """
        # 获取bbox及对应的score
        bbox_infos=dets[:,:8]
        scores=dets[:,8]

        keep=[]
        order=scores.argsort()[::-1]
        nums=len(bbox_infos)
        suppressed=np.zeros((nums), dtype=np.int)

        # 循环遍历
        for i in range(nums):
            idx=order[i]
            if suppressed[idx]==1:
                continue
            keep.append(idx)
            mask_a, area_a = MASK_NMS.get_mask(bbox_infos[idx], mask)
            for j in range(i,nums):
                idx_j=order[j]
                if suppressed[idx_j]==1:
                    continue
                mask_b, area_b = MASK_NMS.get_mask(bbox_infos[idx_j], mask)

                # 获取两个文本的相交面积
                merge_mask=cv2.bitwise_and(mask_a,mask_b)
                area_intersect=cv2.countNonZero(merge_mask)

                #计算MMI
                mmi=MASK_NMS.comput_mmi(area_a,area_b,area_intersect)

                if mmi >= thres:
                    suppressed[idx_j] = 1

        return dets[keep]