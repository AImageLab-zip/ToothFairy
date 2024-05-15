# Global Reference Attention Guided Segmentation (GRADE)
Contribution to the ToothFairy - Cone-Beam Computed Tomography Segmentation Challenge - MICCAI 2023:
by Tomasz Szczepański, Michal K. Grzeszczyk and Przemysław Korzeniowski.

Reproduction:
1. Install environmet using conda and provided environment file: env_config.yml.
2. Download challange dataset and generate pseudo labels using [Deep Label Propagation](https://github.com/AImageLab-zip/alveolar_canal).
We pre-train our models on pseudo-labeled training data and then fine-tune them on data with ground truth labels from [challange dataset](https://toothfairy.grand-challenge.org/dataset/). 
4. Train the following 4 models in order:
    * train_lq.py : coarse segmentation on low resolution dense pseudo-labels -> probability maps - global context reference
    * train_lq_finetune.py : fine tuning on ground truth labels
    * train_grc_pretrain.py : fine segmentation on high resolution dense pseudo-labels, use of gcr as 2ch input
    * train_gcr_finetune.py : fine tuning on ground truth labels - high resolution
5. Paste ensemble of trained models to 'docker/checkpoints'
6. Build docker with build.sh and run to test



ACKNOWLEDGEMENT

The publication was created within the project of the Minister of Science and 
Higher Education "Support for the activity of Centers of Excellence established 
in Poland under Horizon 2020" on the basis of the contract number 
MEiN/2023/DIR/3796  
 
This project has received funding from the European Union’s Horizon 2020 
research and innovation programme under grant agreement No 857533  
 
This publication is supported by Sano project carried out within the 
International Research Agendas programme of the Foundation for Polish 
Science, co-financed by the European Union under the European Regional 
Development Fund 
 
Sano Centre for Computational Medicine, Kraków, Poland 
(https://sano.science/) or Sano Centre for Computational Medicine, Health Informatics Group (HIGS) Team, 
Nawojki 11, 30-072 Kraków, Poland (https://sano.science/).