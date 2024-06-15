#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author:Zhengqing YU
# @contact:yuzheng.qing@163.com
# @file:facialarea.py
# @time:2021/12/29 22:23

import mediapipe as mp
import numpy as np
import cv2
import json
from numpy import mat
import imutils
from codes import json_from_dlib


class FaceSeperate():
    ''' Face seperate based on mediapipe, save 468 procruste json file
     and give facial area map, give facial area 4 kind list(median,
     max,min,sd)'''
    def __init__(self, viimage, tiimage, vijson, tijson, drawviface=True ):
        self.viimage = viimage
        self.tiimage = tiimage
        self.vijson = vijson
        self.tijson = tijson
        self.drawviface = drawviface

    def __ret_x__(self):
        return [self.x]

    def __ret_y__(self):
        return [self.y]

    def __ret_id__(self):
        return [self]

    def __map_max__(self):
        return max(self)

    def __map_min__(self):
        return min(self)

    def __map_std__(self):
        return np.std(self, ddof=1)

    def __map_mean__(self):
        return sum(self) / len(self)

    def __map_pixel__(self):
        return  len(self)

    def mediapipe_VLI(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        # For image input:
        drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=1)
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        image = cv2.imread(self.viimage)
        image = cv2.flip(image, 1)
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        # and print landmarks' id, x, y, z
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if self.drawviface:
                    # Draw landmarks on the image.
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACE_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
                    cv2.namedWindow('MediaPipe FaceMesh', cv2.WINDOW_NORMAL)
                    cv2.imshow('MediaPipe FaceMesh', image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                ih, iw, ic = image.shape
                in_id = np.arange(468).tolist()
                # print(face_landmarks)
                # print(ih,iw,ic)
                id_s = np.array(list(map(FaceSeperate.__ret_id__, in_id)))
                x = np.array(list(map(FaceSeperate.__ret_x__, face_landmarks.landmark[:])))*iw
                y = np.array(list(map(FaceSeperate.__ret_y__, face_landmarks.landmark[:])))*ih
                js = np.concatenate((id_s.astype(int), x.astype(int), y.astype(int)), axis=1)
                with open(self.vijson, "w") as f:
                    json.dump(js.tolist(), f)
                print("The visible light detects the face and loads it into json file!")

    def __transformation_from_points__(self, points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)

        c1 = np.mean(points1, axis=0)  # Centralization
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        s1 = np.std(points1)  # normalized
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = np.linalg.svd(points1.T * points2)
        # The R we seek is in fact the transpose of the one given by U * Vt. This
        # is because the above formulation assumes the matrix goes on the right
        # (with row vectors) where as our solution requires the matrix to be on the
        # left (with column vectors).
        R = (U * Vt).T
        return np.vstack([np.hstack(((s2 / s1) * R,
                                     c2.T - (s2 / s1) * R * c1.T)),
                          np.matrix([0., 0., 1.])])


    def __warp_im__(self, im, M):
        # output_im = np.zeros(dshape, dtype=im.dtype)
        output_im = np.dot(M[:2], np.append(im.T, [np.ones(im.shape[0])], axis=0))
        return output_im.T

    def procrust(self, imagesave=False, imagesavepath=None):
        with open(self.vijson, 'r') as load_f:
            load_mp = json.load(load_f)
            # print(load_mp)

        with open(self.tijson, 'r') as load_f:
            load_dl = json.load(load_f)


        src = np.array(load_mp, np.float32)
        src = src[:, [1, 2]]
        dst = np.array(load_dl['shapes'][1]['points'], np.float32)
        srcselect = src[[0, 61, 291, 17, 130, 243, 463, 359, 34, 264, 4, 94, 152, 367, 138]]
        # ss = srcselect
        srcselect = np.array(srcselect, np.float32)
        dstselect = dst[[49, 48, 50, 51, 36, 39, 42, 45, 0, 16, 30, 33, 8, 12, 4]]
        # sd = dstselect
        dstselect = mat(dstselect)
        srcselect = mat(srcselect)
        rmatrix = FaceSeperate.__transformation_from_points__(self, srcselect, dstselect)  # 得到变换矩阵


        warped_points = FaceSeperate.__warp_im__(self, src, rmatrix)
        # TIjson adds warmed_ Mask new 468 feature point content
        if len(load_dl['shapes']) <= 2: # Eliminate repeated addition
            shapetype = '468features'
            json_from_dlib.write_json(self.tijson, shapetype, points=np.array(warped_points).tolist())
        else:
            pass

        if imagesave:
            if imagesavepath is None:
                print("Please add the saved 468 mesh face path.")
            else:
                print(self.tiimage)
                image = cv2.imread(self.tiimage)
                for (sx, sy) in np.array(warped_points, int):
                    cv2.circle(image, (sx, sy), 2, (0, 0, 255), -1)
                # cv2.imshow("Image", image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(imagesavepath, image)
        else:
            pass
        # cv2.namedWindow('Image', 0)
        # cv2.resizeWindow('Image', 800, 600)
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return warped_points

    def __rect_contains__(self, rect, point):
        if point[0] < rect[0]:
            return False
        elif point[1] < rect[1]:
            return False
        elif point[0] > rect[2]:
            return False
        elif point[1] > rect[3]:
            return False
        return True

    # Draw a point
    def __draw_point__(self, img, p, color):
        cv2.circle(img, p, 2, color)

    # Draw delaunay triangles in
    def __draw_delaunay__(self, img, subdiv, delaunay_color):
        trangleList = subdiv.getTriangleList()
        size = img.shape
        r = (0, 0, size[1], size[0])
        for t in trangleList:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            if (FaceSeperate.__rect_contains__(self, r, pt1)
                    and FaceSeperate.__rect_contains__(self, r, pt1)
                    and FaceSeperate.__rect_contains__(self, r, pt3)):
                cv2.line(img, pt1, pt2, delaunay_color, 1)
                cv2.line(img, pt2, pt3, delaunay_color, 1)
                cv2.line(img, pt3, pt1, delaunay_color, 1)

    def get_delainays(self, warped_points, showdelainay=False):
        # Define window names;
        # Turn on animations while drawing triangles
        # animate = False  # if True will plot points step by step
        # Define colors for drawing
        # delaunary_color = (255, 255, 255)
        # points_color = (0, 0, 255)
        # Read in the image
        # imagepath = './path_and_image_name.png'

        img = cv2.imread(self.tiimage)
        # Keep a copy around
        # img_orig = img.copy()
        # Rectangle to be used with Subdiv2D
        rect = (0, 0, img.shape[1], img.shape[0])

        # Create an instance of Subdiv2d
        subdiv = cv2.Subdiv2D(rect)
        # Create an array of points
        points = np.matrix.tolist(warped_points)
        # Read in the points from a text file
        # with open(r"./mapmodel.txt") as file:
        #    for line in file:
        #        x, y = line.split()
        #        points.append((int(x), int(y)))
        # Insert points into subdiv
        subdiv.insert(points)
        # Show animate
        # if animate:
        #    img_copy = img_orig.copy()
        # Draw delaunay triangles
        #    draw_delaunay(img_copy, subdiv, (255, 255, 255))
        #    cv2.imshow(win_delaunary, img_copy)
        #    cv2.waitKey(100)
        # Draw delaunary triangles
        FaceSeperate.__draw_delaunay__(self,img, subdiv, (255, 255, 255))
        subdiv.getTriangleList()
        # Draw points
        # warped_mask.astype(np.int)
        if showdelainay:
            win_delaunary = "Delaunay Triangulation"
            points = np.matrix.tolist(warped_points.astype(np.int))
            for p in points:
                FaceSeperate.__draw_point__(self, img, tuple(p), (0, 0, 255))
            # Show results
            cv2.imshow(win_delaunary, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return subdiv.getTriangleList()

    def get_standard_delainays(self, warped_points, mapmodel):
        points = np.array(np.matrix.tolist(warped_points))
        return np.hstack((points[mapmodel[0].astype(int)],
                          points[mapmodel[1].astype(int)],
                          points[mapmodel[2].astype(int)]))


    # Basic mask data
    def faceregion_map(self, triangle_list, mapsave):
        vidata = cv2.imread(self.tiimage)
        # Screen out the maximum noise in the image
        vidata = cv2.medianBlur(vidata, 3)
        savearea = []
        origin_mask = np.zeros((vidata.shape[0] * 3, vidata.shape[1] * 3),
                               dtype="uint8")
        i = 0
        for triangle in triangle_list:  #
            mask = np.zeros(vidata.shape[:2], dtype="uint8")  # Mask initialization
            mask = cv2.fillPoly(mask, [triangle.reshape(3, 2).astype(np.int)], (1), 8, 0)
            x = mask.nonzero()
            # area = cv2.bitwise_and(vidata, vidata, mask=mask) The mask takes the value in the area
            are_mean = vidata[x][:, 0].mean()  # get first channel
            are_max = vidata[x][:, 0].max()
            are_min = vidata[x][:, 0].min()
            are_std = vidata[x][:, 0].std()
            are_median = np.median(vidata[x][:, 0])
            mask = imutils.resize(mask, width=mask.shape[1] * 3)
            text = str(i)
            cv2.putText(mask, text, (3 * x[1].mean().astype(np.int), 3 * x[0].mean().astype(np.int)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, (100, 200, 200), 1)
            savearea.append([are_mean, are_max, are_min, are_std, are_median])
            # print(type(are_max), 'are_max'+are_max)
            origin_mask = cv2.add(origin_mask, mask * are_max)
            i = i + 1
        cv2.imshow("mask", origin_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(mapsave, origin_mask)

    def fill_map(self, triangle_list, fill_temp, mapsave, whitecolor=False):
        vidata = cv2.imread(self.tiimage)
        origin_mask = np.zeros((vidata.shape[0], vidata.shape[1]),
                               dtype="uint8")
        i = 0
        for triangle in triangle_list:
            mask = np.zeros(vidata.shape[:2], dtype="uint8")  # Mask initialization
            mask = cv2.fillPoly(mask, [triangle.reshape(3, 2).astype(np.int)], (1), 8, 0)
            # origin_mask = cv2.add(origin_mask, mask * int(fill_temp[i][0]*481))
            #x2 = mask.nonzero()  #de
            #origin_mask[x2] = int(fill_temp[i][0]) #de
            origin_mask = cv2.add(origin_mask, mask * abs(int(fill_temp[i][0])))
            i = i + 1
        if whitecolor:
            np.place(origin_mask, origin_mask == 0, 255)
        cv2.imshow("mask", origin_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(mapsave, origin_mask) # Coloring of face area and image storage
        return origin_mask


    def get_temps(self, triangle_list):
        vidata = cv2.imread(self.tiimage)
        # Screen out the maximum noise in the image
        vidata = cv2.medianBlur(vidata, 3)
        # triangle_list = subdiv.getTriangleList()
        savearea = []
        for triangle in triangle_list:
            mask = np.zeros(vidata.shape[:2], dtype="uint8")  # Mask initialization
            mask = cv2.fillPoly(mask, [triangle.reshape(3, 2).astype(np.int)], (1), 8, 0)
            x = mask.nonzero()
            savearea.append(vidata[x][:, 0])

        # area = cv2.bitwise_and(vidata, vidata, mask=mask)

        meantemp = list(map(FaceSeperate.__map_mean__, savearea))
        maxtemp = list(map(FaceSeperate.__map_max__, savearea))
        mintemp = list(map(FaceSeperate.__map_min__, savearea))
        stdtemp = list(map(FaceSeperate.__map_std__, savearea))
        temparea_pixel = list(map(FaceSeperate.__map_pixel__, savearea))
        return meantemp, maxtemp, mintemp, stdtemp, temparea_pixel





    def facemesh_check_pic(self, facemesh_check_save):
        # read viimage
        image = cv2.imread(self.tiimage)
        # import facemesh json file
        with open(self.tijson, 'r') as load_f:
            load_mp = json.load(load_f)

        src = np.array(load_mp['shapes'][1]['points'],int)
        for (sx, sy) in src:
            cv2.circle(image, (sx, sy), 2, (0, 0, 255), -1)

        cv2.imshow("face_mesh", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(facemesh_check_save, image)  # mesh save

    def get_face_scale(self, model_triangle_list, mapmodel):
        # make faces like the pixel in mapmodel
        vidata = cv2.imread(self.tiimage)  #
        vidata = cv2.medianBlur(vidata, 3)
        with open(self.tijson, 'r') as load_f:
            load_mp = json.load(load_f)
        warped_points = np.matrix(load_mp['shapes'][2]['points'])
        triangle_list = self.get_standard_delainays(warped_points, mapmodel)
        # triangle_list = subdiv.getTriangleList()
        imgsave = np.zeros(vidata.shape, dtype="uint8")
        for triangle1, triangle2 in zip(triangle_list, model_triangle_list):
            img2 = np.zeros(vidata.shape, dtype="uint8")  # Mask initialization black
            tri1 = np.float32([triangle1.reshape(3, 2).astype(np.int)])
            tri2 = np.float32([triangle2.reshape(3, 2).astype(np.int)])
            r1 = cv2.boundingRect(tri1)
            r2 = cv2.boundingRect(tri2)
            # Offset points by left top corner of the
            # respective rectangles
            tri1Cropped = []
            tri2Cropped = []

            for i in range(0, 3):
                tri1Cropped.append(((tri1[0][i][0] - r1[0]), (tri1[0][i][1] - r1[1])))
                tri2Cropped.append(((tri2[0][i][0] - r2[0]), (tri2[0][i][1] - r2[1])))

            # Apply warpImage to small rectangular patches
            img1Cropped = vidata[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
            # Given a pair of triangles, find the affine transform.
            warpMat = cv2.getAffineTransform(np.float32(tri1Cropped), np.float32(tri2Cropped))
            # Apply the Affine Transform just found to the src image
            img2Cropped = cv2.warpAffine(img1Cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT_101)
            # Get mask by filling triangle
            mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0)

            # Apply mask to cropped region
            img2Cropped = img2Cropped * mask

            # Copy triangular region of the rectangular patch to the output image
            img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                    (1.0, 1.0, 1.0) - mask)

            img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3],
                                                             r2[0]:r2[0] + r2[2]] + img2Cropped

            # x = img2.nonzero()
            # mask2 = np.ones(vidata.shape)-np.ones(vidata.shape)*mask  #
            # mask2 = np.ones(vidata.shape)-np.minimum(imgsave,1)
            # (roi, roi, mask=Mask)
            # imgsave = cv2.add(imgsave, np.ndarray(cv2.bitwise_and(img2,img2,mask=mask2),int))
            imgsave = cv2.bitwise_or(imgsave, img2)
        return imgsave

    def get_facemesh_scale(self, model_triangle_list, mapmodel):
        # make faces like the pixel in mapmodel
        # img_path = 'D:/Thermal_project/batch_2021_summer/test/images/1886.png'
        vidata = cv2.imread(self.tiimage)  #
        vidata = cv2.medianBlur(vidata, 3)
        with open(self.tijson, 'r') as load_f:
            load_mp = json.load(load_f)
        warped_points = np.matrix(load_mp['shapes'][2]['points'])
        triangle_list = self.get_standard_delainays(warped_points, mapmodel)
        # triangle_list = subdiv.getTriangleList()
        origin_mask = np.zeros((vidata.shape[0],vidata.shape[1]), dtype='uint8')

        for triangle1, triangle2 in zip(triangle_list, model_triangle_list):
            mask = np.zeros(vidata.shape[:2], dtype="uint8")  # Mask initialization
            mask = cv2.fillPoly(mask, [triangle1.reshape(3, 2).astype(np.int)], (1), 8, 0)
            x = mask.nonzero()
            fill_temp = max(vidata[x][:, 0])
            mask2 = np.zeros((vidata.shape[0],vidata.shape[1]), dtype="uint8")  # Mask initialization
            mask2 = cv2.fillPoly(mask2, [triangle2.reshape(3, 2).astype(np.int)], (1), 8, 0)
            x2 = mask2.nonzero()

            origin_mask[x2] = int(fill_temp)

        return origin_mask


    def crop_face_scale(self, imgsave):

        crop_img = imgsave[96:406, 214:465]
        resize_img = cv2.resize(crop_img, (224, 224))
        # cv2.imshow("cropped", resize_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return resize_img # resize to 224*224