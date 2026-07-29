#!/bin/bash
current_time=`date "+%Y-%m-%d-%H-%M-%S"`
output="./froc-"${current_time}

if [ ! -d ${output} ]; then 
mkdir ${output}
else
rm -rf ${output}
mkdir ${output}
fi

# input_csv="batch_99_faster_rcnn_dconv_c3-c5_r50_fpn_1x_768px_output_nms_0.1.csv"
# input_csv="batch_99_faster_rcnn_dconv_c3-c5_r50_fpn_1x_768px_output_nms_0.1_upp25d-9x48x48.csv"
# input_csv="batch_99_faster_rcnn_dconv_c3-c5_r50_fpn_1x_768px_output_nms_0.1_candidate_dense3d_epoch28.csv"
# input_csv="batch_99_faster_rcnn_dconv_c3-c5_r50_fpn_1x_768px_output_nms_0.1_candidate.csv"
input_csv="batch_99_faster_rcnn_dconv_c3-c5_r50_fpn_1x_7891112969798_768px_output.csv"

# input_csv="retinanet_predict_Yingling_99_0709.csv"
# input_csv="retinanet_predict_Yingling_99_0709_res3d.csv"

# input_csv="AnchorFree_predict_Yingling_99_0709.csv"
input_csv="AnchorFree_predict_Yingling_99_0709_res3d.csv"

cp ./annotations/${input_csv} ${output}
python ./noduleCADEvaluationLUNA16.py ./annotations/lung_data_batch_99.csv ./annotations/annotations_excluded.csv ./annotations/batch_99_dirs.csv ./annotations/${input_csv} ${output}


