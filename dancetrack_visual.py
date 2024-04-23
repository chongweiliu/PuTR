# -*- coding: utf-8 -*-
# @Time     : 2022/6/21 15:42
# @Author   : Liu Chongwei
# @FileName : dancetrack_visual
# @Software : PyCharm

import numpy as np
import cv2
import os

color_list = [[0, 0, 255], [255, 128, 0], [0, 192, 255], [0, 192, 255], [0, 128, 255], [255, 0, 0], [255, 64, 0],
              [64, 0, 255], [0, 255, 64], [192, 0, 255], [255, 0, 255], [0, 255, 192], [0, 255, 255], [255, 192, 0],
              [255, 255, 0], [0, 64, 255], [255, 0, 192], [255, 0, 128], [255, 0, 64], [128, 255, 0], [64, 255, 0],
              [0, 255, 0], [191, 80, 89], [12, 10, 12], [216, 162, 143], [191, 116, 99], [89, 27, 22], [3, 73, 165],
              [10, 55, 114], [104, 165, 87], [216, 214, 203], [114, 54, 41], [3, 89, 165], [5, 41, 63], [3, 165, 60],
              [242, 159, 4], [242, 68, 29], [17, 24, 63], [10, 59, 89], [74, 140, 128], [147, 191, 116],
              [217, 242, 125], [74, 90, 140], [134, 151, 165], [190, 204, 216], [191, 135, 86], [140, 78, 43],
              [253, 2, 2], [243, 46, 12], [249, 86, 16], [253, 139, 46], [248, 166, 43], [243, 190, 31], [248, 225, 15],
              [241, 251, 52], [199, 245, 15], [163, 246, 9], [135, 245, 25], [94, 254, 7], [70, 251, 25], [23, 252, 11],
              [45, 250, 65], [33, 253, 88], [47, 249, 128], [55, 247, 160], [30, 247, 182], [10, 247, 212],
              [48, 251, 251], [16, 217, 252], [26, 181, 248], [40, 152, 243], [5, 104, 252], [56, 104, 248],
              [14, 37, 247], [63, 53, 247], [63, 18, 245], [102, 26, 244], [143, 37, 248], [178, 47, 249],
              [205, 20, 251], [240, 42, 250], [252, 15, 229], [250, 43, 198], [249, 21, 158], [245, 26, 125],
              [252, 10, 83], [249, 40, 71]]


def read(path):

    gt_path_v1 = '/home/lcw/hdd/projects/PuTR/outputs/putr_dancetrack/val_2024-04-23-06-02-24/tracker/' + path + '.txt'
    
    # gt_path_v1 = None
    gt_path_v2 = './YOLOX_outputs/ftv3_dt_val/ftv3_dt_val_test/' + path + '.txt'
    
    gt_path_v2 = './YOLOX_outputs/hs_dt_val/hs_dt_val_val_EGWeightHigh0.0_EGWeightLow0.0_WithLongTermReIDCorrectionFalse_LongTermReIDCorrectionThresh1.0_LongTermReIDCorrectionThreshLow1.0_IoUThresh0.15_ScoreDifInterval1.0_SecScoreDifInterval1.0/' + path + '.txt'
    # gt_path_v2 = f'./datasets/dancetrack/val/{path}/gt/gt.txt'
    
    gt_path_v2 = './YOLOX_outputs/mf_dt_val/mf_dt_val/' + path + '.txt'
    gt_path_v2 = None
    print(gt_path_v1)
    print(gt_path_v2)
    
    img_path = os.path.join(f'./datasets/DanceTrack/val/{path}', 'img1')

    gt_dict_v1 = {}
    if gt_path_v1 is not None:
        with open(gt_path_v1, "r") as f:
            data = f.readlines()
            for i in data:
                i = i.strip('\n').split(',')
                if i[0] in gt_dict_v1.keys():
                    gt_dict_v1[i[0]].append(i)
                else:
                    gt_dict_v1[i[0]] = [i]

    gt_dict_v2 = {}
    if gt_path_v2 is not None:
        with open(gt_path_v2, "r") as f:
            data = f.readlines()
            for i in data:
                i = i.strip('\n').split(',')
                if i[0] in gt_dict_v2.keys():
                    gt_dict_v2[i[0]].append(i)
                else:
                    gt_dict_v2[i[0]] = [i]

    
    thick_line = 1
    thick_font = max(thick_line - 1, 1)
    text_size = cv2.getTextSize('x', 0, fontScale=thick_line / 3., thickness=thick_font)[0]
    
    for i in range(1, 4000):
        gt_list_v1 = gt_dict_v1.get(str(i), [])
        gt_list_v2 = gt_dict_v2.get(str(i), [])
        # if gt_list_v2 is None:
        #     return 0
        img = cv2.imread(img_path + '/' + '0' * (8 - len(str(i))) + str(i) + '.jpg')
        if img is None:
            break
        
        for gt in gt_list_v1:
            xmin, ymin, w, h = gt[2:6]
            xmin, ymin, w, h = int(float(xmin)), int(float(ymin)), int(float(w)), int(float(h))
            cv2.rectangle(img, (xmin, ymin), (xmin + w // 2 if gt_path_v2 is not None else xmin + w, ymin + h), color_list[int(float((gt[1]))) % 80],  thickness=2)
            # cv2.putText(img=img, text=gt[1], org=(xmin + w, ymin + h // 2), fontFace=0, fontScale=thick_line,
            #             color=[255, 255, 0], thickness=thick_font, lineType=cv2.LINE_AA)
            # cv2.rectangle(img=img, pt1=(xmin, ymin), pt2=(xmin + 5, ymin - text_size[1] - 3),
            #               color=[125,125,0], thickness=-1, lineType=cv2.LINE_AA)

        for gt in gt_list_v2:
            xmin, ymin, w, h = gt[2:6]
            xmin, ymin, w, h = int(float(xmin)), int(float(ymin)), int(float(w)), int(float(h))
            cv2.rectangle(img=img, pt1=(xmin + 20, ymin), pt2=(xmin + 25, ymin - text_size[1] - 3), color=[255, 255, 255], thickness=-1,
                          lineType=cv2.LINE_AA)
            cv2.putText(img=img, text=gt[1], org=(xmin + w, ymin), fontFace=0, fontScale=thick_line,
                        color=[0, 255, 0], thickness=thick_font, lineType=cv2.LINE_AA)
            cv2.rectangle(img, (xmin + w // 2 if gt_path_v1 is not None else xmin, ymin), (xmin + w, ymin + h), color_list[int(float(gt[1])) % 80], thickness=2)
            # cv2.rectangle(img, (xmin // 2, ymin// 2), (xmin// 2 + w// 2, ymin// 2 + h// 2), color_list[int(gt[1]) % 80], thickness=2)
        
        cv2.putText(img=img, text=str(i), org=(10, 50), fontFace=0, fontScale=thick_line, color=[0, 255, 0],
                    thickness=thick_font, lineType=cv2.LINE_AA)
        cv2.imshow('img', img)
        if cv2.waitKey(0) == ord('q'):
            break
        elif cv2.waitKey(0) == ord('s'):
            cv2.imwrite(
                f'/home/lcw/hdd/projects/HybirdSORT/exp_data_collections/img_sample/' + f'dancetrack_{path}_' + '0' * (
                            6 - len(str(i))) + str(i) + '.jpg', img)


if __name__ == "__main__":
    read('dancetrack0004')
