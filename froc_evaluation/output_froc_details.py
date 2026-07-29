"""
this module provide abilities to output the orignal images
using the input nod_list
"""
import os
import shutil
import SimpleITK as sitk
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd

DAT_ROOT = r"/home/liuxinglong/data/LUNA/original_lungs"
MAX_OUTPUT_R = 32


def read_dicom_series(case_path):
    """
    read dicom series images
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(case_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image


def read_mhd_singlefile(mhd_path):
    """
    read single mhd file
    """
    image = sitk.ReadImage(mhd_path)
    return image


def world_to_voxel_coord(worldcoord, origin, spacing):
    """
    transform world coord to voxel space
    """
    streched_voxel_coord = np.absolute(worldcoord - origin)
    voxelcoord = streched_voxel_coord / spacing
    return voxelcoord


def lung_trans(img):
    """
    transform the img using given threshold
    :param img: input and output image
    :note : python doesn't support direct ref call, the input
    image is actually the reference of the variable in caller
    :return: No return
    """
    lungwin = np.array([-1200., 600.])
    img = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    img[img < 0] = 0
    img[img > 1] = 1
    img = (img * 255).astype('uint8')
    return img


def output_crop_nodule_details(nod_list, original_image_path, outputpath):
    """
    output nodules with crop size
    """
    if os.path.exists(outputpath):
        shutil.rmtree(outputpath)
    os.makedirs(outputpath)
    radius_scale = 1.5

    from tqdm import tqdm
    # nod_path = os.path.join(DAT_ROOT, "subset" + str(tsubset))
    nod_path = original_image_path
    for idx, nod in enumerate(tqdm(nod_list)):
        ret = nod.transform_inner_data_type()

        img = read_mhd_singlefile(os.path.join(
            nod_path, nod.seriesuid + ".mhd"))
        voxel_coord = world_to_voxel_coord(np.array([nod.coordX, nod.coordY, nod.coordZ]),
                                           img.GetOrigin(), img.GetSpacing())
        voxel_coord = map(int, voxel_coord)
        # print "suid: {0}, voxel_coord: {1}, radius: {2}".format(nod.seriesuid, voxel_coord, int(nod.diameter_mm))

        img_arr = sitk.GetArrayFromImage(img)
        depth, width, height = img_arr.shape
        img_arr = lung_trans(img_arr)

        slice_center_z = voxel_coord[2]
        slice_center_y = voxel_coord[1]
        slice_center_x = voxel_coord[0]
        # make sure this is int since cv2.circle() can only accept integer radiuses
        slice_radius = int(radius_scale * nod.diameter_mm)
        tmp_slice = np.zeros([MAX_OUTPUT_R*2, MAX_OUTPUT_R*2]).astype("uint8")
        for slice_idx in xrange(max(0, slice_center_z - slice_radius), min(slice_center_z + slice_radius, depth)):
            left = max(0, slice_center_y - MAX_OUTPUT_R)
            right = min(slice_center_y + MAX_OUTPUT_R, width)
            top = max(0, slice_center_x - MAX_OUTPUT_R)
            down = min(slice_center_x + MAX_OUTPUT_R, height)
            
            # print "outputting image crop [({0}, {1}),({2}, {3})]".format(left, right, top, down)

            deltaX = right - left
            deltaY = down - top
            tmp_slice[0:deltaX, 0:deltaY] = img_arr[slice_idx, left:right, top:down]
            rgbslice = cv2.cvtColor(tmp_slice, cv2.COLOR_GRAY2RGB)

            cv2.imwrite("{0}/{1}_{2}_{3}.png".format(outputpath,
                                                     nod.seriesuid, idx, slice_idx), rgbslice)


def output_nodule_details(nod_list, original_image_path, outputpath):
    """
    output nodules
    """

    # setup output path
    if os.path.exists(outputpath):
        shutil.rmtree(outputpath)
    os.makedirs(outputpath)

    radius_scale = 1.5

    from tqdm import tqdm
    # nod_path = os.path.join(DAT_ROOT, "subset" + str(tsubset))
    nod_path = original_image_path
    for idx, nod in enumerate(tqdm(nod_list)):
        ret = nod.transform_inner_data_type()

        img = read_mhd_singlefile(os.path.join(
            nod_path, nod.seriesuid + ".mhd"))
        voxel_coord = world_to_voxel_coord(np.array([nod.coordX, nod.coordY, nod.coordZ]),
                                           img.GetOrigin(), img.GetSpacing())
        voxel_coord = map(int, voxel_coord)
        # print "suid: {0}, voxel_coord: {1}, radius: {2}".format(nod.seriesuid, voxel_coord, int(nod.diameter_mm))

        img_arr = sitk.GetArrayFromImage(img)
        depth, width, height = img_arr.shape
        img_arr = lung_trans(img_arr)

        slice_center = voxel_coord[2]
        # make sure this is int since cv2.circle() can only accept integer radiuses
        slice_radius = int(radius_scale * nod.diameter_mm)
        for slice_idx in xrange(max(0, slice_center - slice_radius), min(slice_center + slice_radius, depth)):
            img_slice = img_arr[slice_idx, ...]
            rgbslice = cv2.cvtColor(img_slice, cv2.COLOR_GRAY2RGB)

            # Creates a font
            cv2.putText(rgbslice, "%.6f" % nod.CADprobability,
                        (voxel_coord[0] + slice_radius, voxel_coord[1] + slice_radius),
                        0, 0.5, (0, 255, 0), 1)

            # in-place drawing
            cv2.circle(
                rgbslice, (voxel_coord[0], voxel_coord[1]), slice_radius, (0, 0, 255), 1)

            cv2.imwrite("{0}/{1}_{2}_{3}.png".format(outputpath,
                                                     nod.seriesuid, idx, slice_idx), rgbslice)


def output_nodule_details_tocsv(nod_list, outputpath):
    out_list = []
    column_order = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'radius', 'probability']
    for nod in nod_list:
        out_list.append([nod.seriesuid, nod.coordX, nod.coordY, nod.coordZ, nod.diameter_mm, nod.CADprobability])
    df = pd.DataFrame(out_list, columns=column_order)
    df.to_csv(outputpath + ".csv", index=False)
