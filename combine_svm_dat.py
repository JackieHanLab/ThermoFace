#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author:Zhengqing YU
# @contact:yuzheng.qing@163.com
# @file:combine_svm_dat.py
# @time:2022/12/8 14:13
# -*- coding:utf-8 -*-
# from filesplit.split import Split
# split = Split("./models/dlib_landmark_predictor.dat", "./mergefiles")
# split.bysize(size = 1024*1000*25) # each most 25MB

##combine
from filesplit.merge import Merge
merge = Merge(inputdir = "./mergefiles", outputdir="./models", outputfilename = "dlib_landmark_predictor.dat")
merge.merge()
