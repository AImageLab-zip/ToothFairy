import os
import pathlib

data_path = None #TODO

### Data Paths ###
toothfairy_path = data_path / "ToothFairy_Dataset_V2"

### RAW Data Paths ###
raw_toothfairy_path = toothfairy_path / "RAW"

### Parsed Data Paths ###
parsed_toothfairy_path = toothfairy_path / "PARSED"

### Training Paths ###
project_path = None #TODO
checkpoints_path = project_path / "Checkpoints"
logs_path = project_path / "Logs"
figures_path = project_path / "Figures"
models_path = project_path / "Models"