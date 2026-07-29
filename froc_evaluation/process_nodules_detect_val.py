#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-17 下午6:03
# @Author  : liuxinglong
# @File    : process_nodules_detect.py
# @Description: blabla
import pandas as pd
import math

nodules_detected = "./annotations_val/nodulesDetected.csv"
gt = "./annotations_val/annotations.csv"
output = "./annotations_val/nodulesDetected_labeled.csv"


def main():
    nd_csv = pd.read_csv(nodules_detected, header=None)
    gt_csv = pd.read_csv(gt)

    gt_map = {}
    for gt_idx, gt_data in gt_csv.iterrows():
        if gt_data[0] in gt_map:
            gt_map[gt_data[0]].append([gt_data[1], gt_data[2], gt_data[3], gt_data[4]])
        else:
            gt_map[gt_data[0]] = [[gt_data[1], gt_data[2], gt_data[3], gt_data[4]]]

    # nd_map = {}
    output_csv_list = []
    for nd_idx, nd_data in nd_csv.iterrows():
        # if nd_data[0] in nd_map:
        #     nd_map[nd_data[0]].append([nd_data[1], nd_data[2], nd_data[3], nd_data[4], nd_data[5]])
        # else:
        #     nd_map[nd_data[0]] = [[nd_data[1], nd_data[2], nd_data[3], nd_data[4], nd_data[5]]]
        # for uuid, nd_data in nd_map.iteritems():
        # print "uuid {}, data {}".format(uuid, nd_map)
        uuid = nd_data[0]

        if uuid in gt_map:
            gt_select_data = gt_map[uuid]
        else:
            gt_select_data = None

        if gt_select_data is not None:
            # for d in nd_data:
            bFound = False
            for g in gt_select_data:
                delta = [dd - gg for dd, gg in zip(nd_data[1:4], g[:3])]
                delta_d = nd_data[4] - g[3]
                if delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2] <= delta_d * delta_d:
                    bFound = True
                    output_csv_list.append([nd_data[0], nd_data[1], nd_data[2], nd_data[3], nd_data[4], nd_data[5]] + [1])
                    break
            
            if not bFound:
                output_csv_list.append([nd_data[0], nd_data[1], nd_data[2], nd_data[3], nd_data[4], nd_data[5]] + [0])
        else:
            output_csv_list.append([nd_data[0], nd_data[1], nd_data[2], nd_data[3], nd_data[4], nd_data[5]] + [0])

    df = pd.DataFrame(output_csv_list)
    df.to_csv(output, index=None, header=["seriesuid", "coordX", "coordY", "coordZ", "radius", "probability", "gt"])


if __name__ == '__main__':
    main()
