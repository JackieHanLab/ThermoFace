#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author:Zhengqing YU
# @contact:yuzheng.qing@163.com
# @file:data_mat_load.py
# @time:2021/12/5 19:21
#import numpy as np
#from PIL import Image
#from mlab.releases import latest_release as matlab
import matlab.engine
eng = matlab.engine.start_matlab()
##创建图像存储文件夹


# 运行对应的函数需要先将.m文件置于当前路径下，否则报错找不到
def irdata_png(irdata, pngpath):
    # eg. irdata='D:\Thermal_project\is2test\Images\Main\IR.data',
    #     pngpath='D:/Thermal_project/batch_2021_summer/zip_616-716_gray2/'
    eng.read_data(irdata, pngpath)
    #matlab里头已经按找图片名称存储.png
    return




