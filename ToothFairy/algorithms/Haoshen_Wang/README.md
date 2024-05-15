# Improved-IAN-Segmentation

The 2nd solution to Tooth Fairy: Cone-Beam Computed Tomography (CBCT) Segmentation Challenge. 

## Fine-tuning strategy
The fine_tuning.py script provides a sample implementation of the fine-tuning strategy, showcasing its application on a rudimentary U-Net. However, in practical scenarios, this strategy is executed using the nnU-Net framework.

## Focal Dice loss
The focal Dice loss is defined in loss_function.py and is intended to be utilized in conjunction with the nnU-Net framework.

## Acknowledgements
This code repository refers to [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and [Pytorch Medical Segmentation](https://github.com/MontaEllis/Pytorch-Medical-Segmentation)
