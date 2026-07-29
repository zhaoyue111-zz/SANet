# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from medai_lung_parenchyma_segment import interface
import SimpleITK as sitk
import time
from medai_base_data_define import *
import pandas as pd
import pdb

_ROOT_XIN = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_XIN not in sys.path:
    sys.path.insert(0, _ROOT_XIN)

try:
    import torch
except ImportError:
    torch = None

__all__ = [
    "main",
]


def _run_preprocess(input_image_string):
    """
    调用 cpp_preprocess_dicom_tobuffer，仅记录墙钟耗时。
    """
    t0 = time.time()
    ret, proto_npz = interface.cpp_preprocess_dicom_tobuffer(input_image_string)
    infer_s = time.time() - t0
    return ret, proto_npz, infer_s


def _print_run_summary_zh(df, csv_path):
    """控制台中文汇总。"""
    print("========== 运行汇总 ==========")
    n_all = len(df)
    df_ok = df[df["ret"] == 1]
    n_ok = len(df_ok)
    avg_wall = float(df["infer_s"].mean())

    if n_ok == 0:
        print("墙钟耗时 infer_s：全部 {} 例平均 {:.4f} s；无 ret==1 的成功记录。".format(n_all, avg_wall))
    elif n_ok == n_all:
        print("墙钟耗时 infer_s：共 {} 例且全部成功，平均 {:.4f} s。".format(n_all, avg_wall))
    else:
        avg_ok_wall = float(df_ok["infer_s"].mean())
        print(
            "墙钟耗时 infer_s：全部 {} 例平均 {:.4f} s；其中成功 {} 例平均 {:.4f} s。".format(
                n_all, avg_wall, n_ok, avg_ok_wall
            )
        )

    print("结果已写入：{}".format(csv_path))
    print("==============================")


def main(input_path=None, input_uuid=None, output_path=None):
    interface.DEBUG_CONSOLE_ON = True
    # interface.WRITE_OUTPUT = True

    time_records = []

    params = StringList()
    params.data.append("USE_PPL")
    params.data.append("0.0")
    # params.data.append("1.0")
    # test init
    interface.USE_PPL = 0.0
    interface.initialize_interface()

    interface.cpp_set_default_params(params.SerializeToString())

    # interface.WRITE_OUTPUT = True
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(8)

    if input_path is None:

        # # data_path = '/data1/RJ_project/update0626/1024'
        # data_path = '/data2/肺分割优化/问题数据/肺动脉高压_滤数据/img'
        # output_path = '/data2/肺分割优化/问题数据/肺动脉高压_滤数据/lung_ppl'

        data_path = 'luna25/luna25_images_nii'
        output_path = 'lungseg'

        # data_path = '/data/common/lung/lungSP/heyou/protobuf/raw'
        # output_path = '/data/common/lung/lungSP/heyou/protobuf/raw_out'
        # dataset_name_bucket = ['instance']
        # dataset_name_bucket = os.listdir(data_path)

        os.makedirs(output_path, exist_ok=True)
        dataset_name_bucket = sorted(os.listdir(data_path))[:]
        for dataset_name in dataset_name_bucket:
            try:
                suuid = dataset_name.replace('.nii.gz', '')
                output_nii_path = os.path.join(output_path, "{}.nii.gz".format(suuid))
                if os.path.exists(output_nii_path):
                    print("skip {}, output already exists: {}".format(suuid, output_nii_path))
                    continue
                # if not os.path.exists('/data2/肺叶分割优化/big_cancer_mask/'+dataset_name):
                #     continue
                print("*" * 30)
                print("processing {}".format(dataset_name))
                if dataset_name.endswith('.nii.gz'):
                    simage = sitk.ReadImage('{}/{}'.format(data_path, dataset_name))

                else:
                    simage, suuid = load_image('{}/{}'.format(data_path, dataset_name))
                # sitk.WriteImage(simage, data_path + '/' + dataset_name + '.nii.gz')
                dataset_name = suuid
                if simage is None:
                    print("loading image failed")
                    continue

                print("image size {}, spacing {}".format(simage.GetSize(), simage.GetSpacing()))
                print("sitk pixel type {}".format(simage.GetPixelIDTypeAsString()))

                # print('*'*60)
                # print('pro preprocess_dicom_tomask')
                # print('*'*60)
                #
                # ##
                # ret_code, rle_mask = interface.preprocess_dicom_tomask(simage)
                # if ret_code==1:
                #     mask = rleimage2nparray(rle_mask)
                #     smask = sitk.GetImageFromArray(mask)
                #     smask.CopyInformation(simage)
                #     sitk.WriteImage(smask, "{}/{}.nii.gz".format(output_path, dataset_name))
                #     print('mask shape {}'.format(np.shape(mask)))

                print('*' * 60)
                print('pro cpp_predict_newimage_custom')
                print('*' * 60)
                input_image_string = simage2dicomimage(simage, suuid)
                # ret, rle_mask = interface.preprocess_dicom_tomask(simage)
                # pdb.set_trace()
                ret, proto_npz, infer_s = _run_preprocess(input_image_string)
                record = {
                    "file": suuid,
                    "infer_s": float(infer_s),
                    "cost_time": float(infer_s),
                    "ret": int(ret),
                    "lrlung_nonzero": None,
                }

                print("predict done, wall_s {:.4f}, ret {}".format(infer_s, ret))

                if ret == 1:
                    print('success')
                    try:
                        os.makedirs(output_path, exist_ok=True)
                        raw_path = os.path.join(output_path, "{}.raw".format(suuid))
                        # with open(raw_path, "wb") as f:
                        #     f.write(proto_npz if proto_npz is not None else b"")
                    except Exception as ex:
                        print("failed to save proto_npz raw for {}, err: {}".format(suuid, ex))
                    exchange_image = ExchangeImage()
                    exchange_image.ParseFromString(proto_npz)
                    npz_file_contents = exchangeimage2npzdict(exchange_image)
                    lrlung_mask = npz_file_contents['raw_lrmask']
                    record["lrlung_nonzero"] = int(np.count_nonzero(lrlung_mask))
                    mask = sitk.GetImageFromArray(lrlung_mask)
                    mask.CopyInformation(simage)
                    sitk.WriteImage(mask, '{}/{}.nii.gz'.format(output_path, dataset_name))
                time_records.append(record)
            except Exception as ex:
                print("ERROR while processing {}: {}".format(dataset_name, repr(ex)))

                try:
                    record = {
                        "file": dataset_name,
                        "infer_s": None,
                        "cost_time": None,
                        "ret": -999,
                        "lrlung_nonzero": None,
                        "status": "exception",
                        "error": repr(ex),
                    }
                    time_records.append(record)
                except Exception:
                    pass

                continue

    try:
        if time_records:
            df = pd.DataFrame(time_records)
            csv_dir = output_path if output_path else os.getcwd()
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, "infer_times.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            _print_run_summary_zh(df, csv_path)
    except Exception as ex:
        print("failed to save infer_times.csv, err: {}".format(ex))

    # test destroy interface
    ret = interface.finalize_interface()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        path = sys.argv[1]
        uuid = sys.argv[2]
        main(input_path=path, input_uuid=uuid, output_path=None)

    elif len(sys.argv) == 4:
        path = sys.argv[1]
        uuid = sys.argv[2]
        outpath = sys.argv[3]
        main(input_path=path, input_uuid=uuid, output_path=outpath)

    else:
        main()