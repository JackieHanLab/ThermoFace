# This is the main pipeline script.

# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from codes import dlib_predict_thermal, is2_process
from codes.FAS.facialarea import FaceSeperate
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser(description='IS2 data input folder.')
parser.add_argument('-i', '--inputdata', type=str, default='test_data')

args = parser.parse_args()



if __name__ == '__main__':
    basicpath = './data/'    # origin path
    sep = args.inputdata         # sep batch path'
    is2_process.mkdir(path=basicpath + '{}_result/'.format(sep) + r'png/')
    is2_process.is2_process(rawpath=basicpath + sep + r'/',
                            path=basicpath + r'{}_result/zip/'.format(sep))
    # create json document
    basicpath = basicpath + '{}_result/'.format(sep)
    is2_process.mkdir(path=basicpath + r'json')

    dlib_predict_thermal.dlib_pre_thermal(images=basicpath + r'png/',
                                          jsonpath=basicpath + r'json/')

    # FAS face area seperate
    imageids = list(os.listdir(basicpath + 'zip/'))
    mapmodel = np.loadtxt("./models/mapmodel.txt", dtype=int, delimiter=",")
    # create blank json document
    is2_process.mkdir(path=basicpath + r'vijson/')
    vijson_dir = basicpath + r'vijson/'
    # create facemesh save document
    is2_process.mkdir(path=basicpath + r'facemesh/')
    facemesh_check_dir = basicpath + r'facemesh/'
    basepath = basicpath + 'zip/'
    for ind, imageid in enumerate(imageids, 1):
        print("[INFO] Processing image: {}/{}".format(ind, len(imageids)))
        viimage = basepath + '{}/Images/Main/028001E0.jpg'.format(imageid)   # batch2&3 0A000780.jpg
        tiimage = basepath.replace('zip', 'png') + '{}.png'.format(imageid)
        vijson = vijson_dir + "{}.json".format(imageid)
        tijson = basepath.replace('zip', 'json') + '{}.json'.format(imageid)
        facemesh_check_save = facemesh_check_dir + '{}.png'.format(imageid)
        facetest = FaceSeperate(viimage, tiimage, vijson, tijson, drawviface=False)
        # detect face mesh in viimage save mesh json into vijson
        facetest.mediapipe_VLI()
        # get procruste changed 468 matrix
        if not os.path.exists(vijson):  # exclude unqualitied viimages
            continue
        warped_points = facetest.procrust(imagesave=True, imagesavepath=facemesh_check_save)
        # get triangle list and also show the triangle in each feed in tiimage
        triangle_list = facetest.get_standard_delainays(warped_points, mapmodel)
        # get back every types of temperature in each facial area
        meantemp, maxtemp, mintemp, stdtemp, temparea_pixel = facetest.get_temps(triangle_list)
        data_meantemp = DataFrame({imageid: meantemp})
        data_maxtemp = DataFrame({imageid: maxtemp})
        data_mintemp = DataFrame({imageid: mintemp})
        data_stdtemp = DataFrame({imageid: stdtemp})
        data_temparea_pixel = DataFrame({imageid: temparea_pixel})
        if ind == 1:
            data_save_meantemp = data_meantemp
            data_save_maxtemp = data_maxtemp
            data_save_mintemp = data_mintemp
            data_save_stdtemp = data_stdtemp
            data_save_temparea_pixel = data_temparea_pixel
        else:
            data_save_meantemp = pd.concat([data_save_meantemp, data_meantemp], axis=1)
            data_save_maxtemp = pd.concat([data_save_maxtemp, data_maxtemp], axis=1)
            data_save_mintemp = pd.concat([data_save_mintemp, data_mintemp], axis=1)
            data_save_stdtemp = pd.concat([data_save_stdtemp, data_stdtemp], axis=1)
            data_save_temparea_pixel = pd.concat([data_save_temparea_pixel, data_temparea_pixel], axis=1)

    # create average document
    is2_process.mkdir(path=basicpath + r'tempe/')
    tempe_dir = basicpath + r'tempe/'
    data_save_meantemp.to_csv(tempe_dir + 'mean.csv')
    data_save_maxtemp.to_csv(tempe_dir + 'max.csv')
    data_save_mintemp.to_csv(tempe_dir + 'min.csv')
    data_save_stdtemp.to_csv(tempe_dir + 'std.csv')
    data_save_temparea_pixel.to_csv(tempe_dir + 'area_pixel.csv')



