#echo nnUNetv2_Best
#nnUNetv2_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0/test/imagesTs -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter1_data_filter/val_infer_SDA/nnunetv2/best -d 106 -c 3d_fullres -f all -chk checkpoint_best.pth
#
#echo nnUNetv2_1000
#nnUNetv2_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0/test/imagesTs -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter1_data_filter/val_infer_SDA/nnunetv2/1000 -d 106 -c 3d_fullres -f all -chk 999.pth
#
#echo nnUNetv2_666
#nnUNetv2_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0/test/imagesTs -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter1_data_filter/val_infer_SDA/nnunetv2/666 -d 106 -c 3d_fullres -f all -chk 666.pth
#
#echo nnUNetv2_333
#nnUNetv2_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0/test/imagesTs -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter1_data_filter/val_infer_SDA/nnunetv2/333 -d 106 -c 3d_fullres -f all -chk 333.pth





echo nnUNetv2_Best
nnUNetv2_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter1_data_filter/img_unlabeled -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter1_data_filter/pseudo_infer/nnunetv2/best -d 106 -c 3d_fullres -f all -chk checkpoint_best.pth

echo nnUNetv2_1000
nnUNetv2_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter1_data_filter/img_unlabeled -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter1_data_filter/pseudo_infer/nnunetv2/1000 -d 106 -c 3d_fullres -f all -chk 999.pth

echo nnUNetv2_666
nnUNetv2_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter1_data_filter/img_unlabeled -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter1_data_filter/pseudo_infer/nnunetv2/666 -d 106 -c 3d_fullres -f all -chk 666.pth

echo nnUNetv2_333
nnUNetv2_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter1_data_filter/img_unlabeled -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter1_data_filter/pseudo_infer/nnunetv2/333 -d 106 -c 3d_fullres -f all -chk 333.pth

#echo nnUNetv2_Best
#nnUNetv2_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0_data_filter/image_bad -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0_data_filter/pseudo_infer/nnunetv2/best -d 105 -c 3d_fullres -f all -chk checkpoint_best.pth
#
#echo nnUNetv2_1000
#nnUNetv2_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0_data_filter/image_bad -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0_data_filter/pseudo_infer/nnunetv2/1000 -d 105 -c 3d_fullres -f all -chk 999.pth
#
#echo nnUNetv2_666
#nnUNetv2_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0_data_filter/image_bad -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0_data_filter/pseudo_infer/nnunetv2/666 -d 105 -c 3d_fullres -f all -chk 666.pth
#
#echo nnUNetv2_333
#nnUNetv2_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0_data_filter/image_bad -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0_data_filter/pseudo_infer/nnunetv2/333 -d 105 -c 3d_fullres -f all -chk 333.pth


#echo nnUNetv2_Best
#nnUNetv2_predict -i /media/ps/lys_ssd/nnunetv2_data/nnUNet_raw_data_base/Dataset103_IAN_Dense_Val/imagesTr -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter1/fine_label_img_filter/nnunetv2 -d 103 -c 3d_fullres -f all -chk checkpoint_best.pth
