#echo nnUNet_Best
#nnUNet_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0/test/imagesTs -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter1_data_filter/val_infer_SDA/nnunet/best -tr nnUNetTrainerV2 -t 106 -m 3d_fullres -f all -chk model_best
#
#echo nnUNet_1000
#nnUNet_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0/test/imagesTs -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter1_data_filter/val_infer_SDA/nnunet/1000 -tr nnUNetTrainerV2 -t 106 -m 3d_fullres -f all -chk 999
#
#echo nnUNet_666
#nnUNet_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0/test/imagesTs -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter1_data_filter/val_infer_SDA/nnunet/666 -tr nnUNetTrainerV2 -t 106 -m 3d_fullres -f all -chk 666
#
#echo nnUNet_333
#nnUNet_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0/test/imagesTs -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter1_data_filter/val_infer_SDA/nnunet/333 -tr nnUNetTrainerV2 -t 106 -m 3d_fullres -f all -chk 333




echo nnUNet_Best
nnUNet_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter2_data_filter/image_unlabeled -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter2_data_filter/pseudo_infer/best -tr nnUNetTrainerV2 -t 109 -m 3d_fullres -f all -chk model_best

echo nnUNet_1000
nnUNet_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter2_data_filter/image_unlabeled -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter2_data_filter/pseudo_infer/1000 -tr nnUNetTrainerV2 -t 109 -m 3d_fullres -f all -chk 999

echo nnUNet_666
nnUNet_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter2_data_filter/image_unlabeled -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter2_data_filter/pseudo_infer/666 -tr nnUNetTrainerV2 -t 109 -m 3d_fullres -f all -chk 666

echo nnUNet_333
nnUNet_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter2_data_filter/image_unlabeled -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter2_data_filter/pseudo_infer/333 -tr nnUNetTrainerV2 -t 109 -m 3d_fullres -f all -chk 333

#echo nnUNet_Best
#nnUNet_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0_data_filter/image_bad -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0_data_filter/pseudo_infer/nnunet/best -tr nnUNetTrainerV2 -t 105 -m 3d_fullres -f all -chk model_best
#
#echo nnUNet_1000
#nnUNet_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0_data_filter/image_bad -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0_data_filter/pseudo_infer/nnunet/1000 -tr nnUNetTrainerV2 -t 105 -m 3d_fullres -f all -chk 999
#
#echo nnUNet_666
#nnUNet_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0_data_filter/image_bad -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0_data_filter/pseudo_infer/nnunet/666 -tr nnUNetTrainerV2 -t 105 -m 3d_fullres -f all -chk 666
#
#echo nnUNet_333
#nnUNet_predict -i /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0_data_filter/image_bad -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter0_data_filter/pseudo_infer/nnunet/333 -tr nnUNetTrainerV2 -t 105 -m 3d_fullres -f all -chk 333


#echo nnUNet_Best
#nnUNet_predict -i /media/ps/lys_ssd/nnunetv2_data/nnUNet_raw_data_base/Dataset103_IAN_Dense_Val/imagesTr -o /media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter1/fine_label_img_filter/nnunet -tr nnUNetTrainerV2 -t 103 -m 3d_fullres -f all -chk model_best
