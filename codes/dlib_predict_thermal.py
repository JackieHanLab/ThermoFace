#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author:Zhengqing YU
# @contact:yuzheng.qing@163.com
# @file:dlib_predict_thermal.py
# @time:2022/1/13 21:49
"""
python dlib_predict_image.py --images D:/Thermal_project/batch_2021_summer/test/images/
--models  models/ --upsample 0 -j D:/Thermal_project/batch_2021_summer/test/json/
-o D:/Thermal_project/batch_2021_summer/test/output/

"""
# import the necessary packages
from imutils import face_utils
from imutils import paths
import imutils
import dlib
import cv2
import os
from codes import json_from_dlib

def dlib_pre_thermal(images, jsonpath,
                     models='models/', showres=False, output=None):
    # load the face detector (HOG-SVM)

    print("[INFO] loading dlib thermal face detector...")
    detector = dlib.simple_object_detector(os.path.join(models, "dlib_face_detector.svm"))

    # load the facial landmarks predictor
    print("[INFO] loading facial landmark predictor...")
    predictor = dlib.shape_predictor(os.path.join(models, "dlib_landmark_predictor.dat"))

    # grab paths to the images
    imagePaths = list(paths.list_files(images))
    # print("Error: 没有找到文件或读取文件失败")
    # loop over the images
    for ind, imagePath in enumerate(imagePaths, 1):
        print("[INFO] Processing image: {}/{}".format(ind, len(imagePaths)))
        # load the image
        image = cv2.imread(imagePath)
        # resize the image
        img_orgin = image.shape[1]
        zoom_p = image.shape[1] / 300
        image = imutils.resize(image, width=300)
        # copy the image
        image_copy = image.copy()
        # convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        # detect faces in the image
        rects = detector(image, upsample_num_times=1)
        # print(rects.shape)
        if len(rects) != 1:
            Max = 0  # set for bigger rec
            for i in rects:
                (x, y, w, h) = face_utils.rect_to_bb(i)
                if h > Max:
                    Max = h  # 寻找最大的宽度的rec
                    # print(h)
                    rect = i
            rects_one = dlib.rectangles()
            rects_one.append(rect)
        else:
            rects_one = rects
        print(rects_one)
        for rect in rects_one:  # one means only one biggest face can be written into json
            # convert the dlib rectangle into an OpenCV bounding box and
            # draw a bounding box surrounding the face

            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # save xs,ys,xe,ye in json file
            points = [[x * zoom_p, y * zoom_p], [x * zoom_p + w * zoom_p, y * zoom_p + h * zoom_p]]
            # set json file name;
            imagename = imagePath.split('/')[-1]
            jsonfile = os.path.join(jsonpath, imagename.replace('.png', '.json'))
            json_from_dlib.initial_json(jsonfile, imgpath=imagename,
                                        imgh=image.shape[0] * zoom_p, imgw=image.shape[1] * zoom_p)
            # save with label='face';
            shapetype = 'face'
            # read jsonfile and add face rectangle landmarks
            json_from_dlib.write_json(jsonfile, shapetype, points)
            # 在图像上提取
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # predict the location of facial landmark coordinates,
            # then convert the prediction to an easily parsable NumPy array
            shape = predictor(image, rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y)-coordinates from our dlib shape
            # predictor model draw them on the image
            ftpoints = []
            for (sx, sy) in shape:
                cv2.circle(image_copy, (sx, sy), 2, (0, 0, 255), -1)
                # save feature points json
                ftpoints.append([sx * zoom_p, sy * zoom_p])
            json_from_dlib.write_json(jsonfile, 'features', ftpoints)
        if showres:
            # show the image
            if output is None:
                print("Please add the saved 468 mesh face path.")
            else:
                cv2.imshow("Image", image_copy)
                image_copy = imutils.resize(image_copy, width=img_orgin)
                cv2.imwrite(output + imagePath.split('/')[-1], image_copy)
                key = cv2.waitKey(0) & 0xFF
        else:
            pass



