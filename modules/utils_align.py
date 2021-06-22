# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
from numpy.linalg import inv, norm
from numpy.linalg import matrix_rank as rank

import cv2
import torch

class AlignFace(object):
    def __init__(self):
        self.REFERENCE_FACIAL_POINTS_96x112 = [
            [30.29459953, 51.69630051],
            [65.53179932, 51.50139999],
            [48.02519989, 71.73660278],
            [33.54930115, 92.3655014],
            [62.72990036, 92.20410156]
        ]
        self.REFERENCE_FACIAL_POINTS_112x112 = [
            [38.29459953, 51.69630051],
            [73.53179932, 51.50139999],
            [56.02519989, 71.73660278],
            [41.54930115, 92.3655014 ],
            [70.72990036, 92.20410156]
        ]
    def __call__(self, frame, facial_5pts, boxes, align_type):
#        print('facial_5pts_ori:', facial_5pts)
#        print('self.REFERENCE_FACIAL_POINTS_112x112:', self.REFERENCE_FACIAL_POINTS_112x112)
#        facial_5pts = np.reshape(facial_5pts, (2, -1))
#        print('facial_5pts:', facial_5pts)
        facial = []
        x = facial_5pts[::2]
        y = facial_5pts[1::2]
        facial.append(x)
        facial.append(y)
        dst_img = self.warpAndCrop(frame, facial, boxes, (112, 112), align_type)
        return dst_img
    def warpAndCrop(self, src_img, facial_pts, boxes, crop_size, align_type):
        reference_pts = self.REFERENCE_FACIAL_POINTS_112x112
        ref_pts = np.float32(reference_pts)
        ref_pts_shp = ref_pts.shape
        if ref_pts_shp[0] == 2:
            ref_pts = ref_pts.T
        src_pts = np.float32(facial_pts)
        src_pts_shp = src_pts.shape
        if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
            raise Exception('facial_pts.shape must be (K,2) or (2,K) and K>2')
        if src_pts_shp[0] == 2:
            src_pts = src_pts.T
        if src_pts.shape != ref_pts.shape:
            raise Exception('facial_pts and reference_pts must have the same shape: {} vs {}'.format(src_pts.shape, ref_pts.shape) )
        tfm = self.getAffineTransform(src_pts, ref_pts)
        face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))
        if align_type == 'warp_crop':
            points = np.array([[boxes[0], boxes[1]], [boxes[2], boxes[3]]])
            warp_points = cv2.transform(np.reshape(points, (points.shape[0],1,2)), tfm)
            min_x = int(warp_points[0][0][0]) if int(warp_points[0][0][0]) > 0 else 0
            max_x = int(warp_points[1][0][0]) if int(warp_points[1][0][0]) > 0 else 0
            min_y = int(warp_points[0][0][1]) if int(warp_points[0][0][1]) > 0 else 0
            max_y = int(warp_points[1][0][1]) if int(warp_points[1][0][1]) > 0 else 0
            face_img = face_img[min_y:max_y, min_x:max_x]
            face_img = cv2.resize(face_img, (crop_size[0], crop_size[1]))

        return face_img
    def getAffineTransform(self, uv, xy):
        options = {'K': 2}
        trans1, trans1_inv = self.findNonreflectiveSimilarity(uv, xy, options)
        xyR = xy.copy()
        xyR[:, 0] = -1 * xyR[:, 0]
        trans2r, trans2r_inv = self.findNonreflectiveSimilarity(uv, xyR, options)
        TreflectY = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        trans2 = np.dot(trans2r, TreflectY)

        xy1 = self.tformfwd(trans1, uv)
        norm1 = norm(xy1 - xy)
        xy2 = self.tformfwd(trans2, uv)
        norm2 = norm(xy2 - xy)

        if norm1 <= norm2:
            trans = trans1
        else:
            trans2_inv = inv(trans2)
            trans = trans2
        cv2_trans = trans[:, 0:2].T
        return cv2_trans
    def findNonreflectiveSimilarity(self, uv, xy, options=None):
        options = {'K': 2}
        K = options['K']
        M = xy.shape[0]
        x = xy[:, 0].reshape((-1, 1))
        y = xy[:, 1].reshape((-1, 1))
        tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
        tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
        X = np.vstack((tmp1, tmp2))
        u = uv[:, 0].reshape((-1, 1))
        v = uv[:, 1].reshape((-1, 1))
        U = np.vstack((u, v))

        X_tensor = torch.from_numpy(X.astype('float32'))
        U_tensor = torch.from_numpy(U.astype('float32'))
        if rank(X) >= 2 * K:
            r, _ = torch.lstsq(U_tensor, X_tensor)
            r = r[:X_tensor.shape[1]]
            r = r.numpy()
            r = np.squeeze(r)
        else:
            raise Exception('cp2tform:twoUniquePointsReq')
        sc = r[0]
        ss = r[1]
        tx = r[2]
        ty = r[3]

        Tinv = np.array([
            [sc, -ss, 0],
            [ss, sc, 0],
            [tx, ty, 1]
        ])
        Tinv.astype('float64')
        T = inv(Tinv)
        T[:, 2] = np.array([0, 0, 1])
        return T, Tinv
    def tformfwd(self, trans, uv):
        uv = np.hstack((
            uv, np.ones((uv.shape[0], 1))
        ))
        xy = np.dot(uv, trans)
        xy = xy[:, 0:-1]
        return xy
    def drawBoxes(self, img, imgname,res):
        write_path = '../../attendance/web/static/results/{}/{}/'.format( time.strftime("%Y%m%d", time.localtime()),res )
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        cv2.imwrite('{}{}_aligned.jpg'.format(write_path, imgname), img)
        return '/static/results/{}/{}/{}_aligned.jpg'.format(time.strftime("%Y%m%d", time.localtime()),res,imgname)
    def drawBoxes1(self, frame, imgname, boxes, points):

#        boxes = self.d['boxes']
#        points = self.d['points']
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        for i in range(x1.shape[0]):
            rightEyeCenter = (points[i][0], points[i][5])
            leftEyeCenter = (points[i][1], points[i][6])
            p3 = (points[i][2], points[i][7])
            p4 = (points[i][3], points[i][8])
            p5 = (points[i][4], points[i][9])
            cv2.rectangle(frame, (int(x1[i]), int(y1[i])),(int(x2[i]), int(y2[i])), (0, 0, 255), 3)
            #cv2.circle(frame, rightEyeCenter, 2, (0, 0, 255), 3)
            #cv2.circle(frame, leftEyeCenter, 2, (0, 255, 0), 3)
            #cv2.circle(frame, p3, 2, (255, 0, 0), 3)
            #cv2.circle(frame, p4, 2, (255, 255, 0), 3)
            #cv2.circle(frame, p5, 2, (0, 255, 255), 3)
            write_path = '../../attendance/web/static/results/{}/src/'.format( time.strftime("%Y%m%d", time.localtime()) )
            if not os.path.exists(write_path):
                os.makedirs(write_path)
            cv2.imwrite('{}{}.jpg'.format(write_path, imgname), frame)
            return '{}{}.jpg'.format(write_path, imgname)


if __name__ == '__main__':
 #   vca = syszuxav.SYSZUXCamera("",16)
    alignFace = AlignFace()
    frame_num = 0
    while True:
        frame = np.array(vca.decodeJpg())
        if frame.shape[0] == 0:
            continue
        frame_num += 1
        rc = mtcnn.process(frame)
        if not rc:
            continue
        faceimg = frame.copy()
        mtcnn.drawBoxes(frame, frame_num)
        points = mtcnn.d['points']
        if points.shape[0] > 0:
            for i in range(points.shape[0]):
                #gemfield.org: WARNING! THE IMG WILL CONTAIN THE MARK drew earlier!
                alignFace.drawBoxes(alignFace(faceimg, points[i,:]), frame_num,"localtest")
