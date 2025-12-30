# NexToU_ToothFairy2

This repository contains the **5th-place solution** for the **ToothFairy2 Challenge** — a competition focused on multi-structure segmentation in CBCT volumes.  

1. **Environment Setup**  
   ```bash
   unzip nnUNet_hti_nnUNet_v2.5_ToothFairy2.zip
   cd nnUNet_hti_nnUNet_v2.5_ToothFairy2
   pip install -e .
   ```

2. **Build Docker Image**  
   ```bash
   bash build.sh
   ```

3. **Run Inference**  
   ```bash
   bash test.sh
   ```

4. **Export Model**  
   ```bash
   bash export.sh
   ```

**Core Methodology**  
- **Network Architecture**: Based on **[NexToU](https://github.com/PengchengShi1220/NexToU)**.  
- **Loss Function**: Uses hierarchical semantic loss from **[fractal-softmax](https://github.com/PengchengShi1220/fractal-softmax)**.  
- **Framework**: Implemented using the **nnU-Net** framework.  

**Acknowledgments**  
- **[nnU-Net](https://github.com/MIC-DKFZ/nnUNet)** — Base framework.
