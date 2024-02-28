import SimpleITK as sitk
import os
import numpy as np

p_id = "P1"
p = f"/home/tf/ToothFairy_data/ToothFairy_Dataset/Dataset/{p_id}/data.npy"
p_out = f"test/images/cbct/{p_id}.mha"

mask = np.load(p)
mask_sitk = sitk.GetImageFromArray(mask)
# input_array = sitk.GetArrayFromImage(mask_sitk)
# input_array = input_array.astype(np.float32)
# np.save(p_out, input_array)
writer = sitk.ImageFileWriter()
writer.SetFileName(p_out)
writer.Execute(mask_sitk)
