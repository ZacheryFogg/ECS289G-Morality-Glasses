#!/bin/bash

# Initialize the conda environment 
eval "$(conda shell.bash hook)"

conda activate torch-cuda12.4

# Run models 
python train_ethics_model.py

python train_one_model.py base_model 30

python train_one_model.py lora 30

python train_one_model.py adapter 30

python train_one_model.py prefix 30

python train_one_model.py base_model 60

python train_one_model.py lora 60

python train_one_model.py adapter 60

python train_one_model.py prefix 60

python train_one_model.py base_model 300

python train_one_model.py lora 300

python train_one_model.py adapter 300

python train_one_model.py prefix 300

python train_one_model.py base_model 600

python train_one_model.py lora 600

python train_one_model.py adapter 600

python train_one_model.py prefix 600

python train_one_model.py base_model 900

python train_one_model.py lora 900

python train_one_model.py adapter 900

python train_one_model.py prefix 900

# Deactivate conda env 
conda deactivate