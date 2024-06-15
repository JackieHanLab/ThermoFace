#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author:Zhengqing YU
# @contact:yuzheng.qing@163.com
# @file:is2_process.py.py
# @time:2021/11/29 16:04
##函数调用创建文件夹、重命名文件夹解压文件夹、
import os  # 引入模块
import shutil
import re
import zipfile
from codes import data_mat_load


def mkdir(path):
    path = path.strip()  # Remove first space
    path = path.rstrip("\\")  # remove \
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' created successfully')
        return True
    else:
        print(path + ' directory already exists')
        return False


def copy_files(rawpath,path):
    p= r'.+\.IS2'  # r'\w+\.IS2'#Define regular expressions
    for  foldName,subfolders,filenames in os.walk(rawpath):
           for filename in filenames:
              m=re.match(p,filename)

              if m:
                  newfold_name = filename.replace('.IS2', '')
                  if os.path.exists(os.path.join(path, newfold_name)):
                      continue
                  else:
                      new_name = filename.replace('.IS2', '.zip')  # turn .is2 into .zip
                      shutil.copyfile(os.path.join(foldName, filename), os.path.join(path, new_name))
                      mkdir(os.path.join(path, newfold_name))
                      z = zipfile.ZipFile(os.path.join(path, new_name), 'r')
                      z.extractall(path=os.path.join(path, newfold_name))
                      z.close()
                      os.remove(os.path.join(path, new_name))  # Delete the extracted files in the current folder

              else:
                  continue              #
    print('Copy and unzip finished!')

def is2_process(rawpath ,path):
    mkdir(path)
    copy_files(rawpath, path)
    mkdir(path.replace('zip_', 'png_'))
    for i in os.listdir(path):
        data_mat_load.irdata_png(path + i + '/Images/Main/IR.data',
                                 path.replace('zip', 'png'))

    print('Finished is2 preprocess and get png file.')













