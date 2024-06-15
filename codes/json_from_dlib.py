#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author:Zhengqing YU
# @contact:yuzheng.qing@163.com
# @file:json_from_dlib.py
# @time:2021/12/15 20:20
import json
# create json file with "imagePath", "imageHeight", "imageWidth","shapes"
# initialize json file for each input is2 image
def initial_json(jsonpath,imgpath,imgh,imgw):
     x = {
          "imagePath": imgpath,
          "imageHeight": imgh,
          "imageWidth": imgw,
          "shapes":[]
     }
     with open(jsonpath, 'w', encoding='utf-8') as f2:
         json.dump(x,f2)

     return

# add facial area rectangle feature points
# add other facial feature points


def write_json(jsonfile,shapetype,points):
     #首先读取已有的json文件中的内容
    with open(jsonfile, 'r') as f:
        load_dict = json.load(f)
        d = {'label': shapetype, 'points': points}
        # 将新传入的dict对象追加至list中
        load_dict['shapes'].append(d)
        #load_dict["c"] = 3
    #将追加的内容与原有内容写回（覆盖）原文件
    with open(jsonfile, 'w', encoding='utf-8') as f2:
        json.dump(load_dict, f2, indent=1) # indent to check whether json format or not;
