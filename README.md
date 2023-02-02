# constrained-lolbo-for-antibodies
Code to run constrained lolbo for ab design (default is VAE pre-trained on OAS IGHG heavy chains)

# Getting Started 

## Environment requirements 
pytorch 
fcd_torch 
gpytorch 
numpy 
botorch 
pytorch-lightning
pymolPy3
pymol 
igfold 
fire
pyrosetta 
git-lfs 
pdb-tools 
wandb 
pytorch3d
rdkit
rdkit-pypi
prody 

## Set default constants and data paths

Open constants.py and change constants as desired 
(paths to initialization data, antigen pdb file, full parental sequence, etc.)

## Initialization data 
The current version of init_data.csv has sequences and scores just to show the desired csv format for the initial dataset 

Replace init_data.csv with the set of the sequences and all associated scores you want to use to initialize the optimization run 

Use the same headings as are currently in the init_data.csv file 

# Example Command to run optimization: 

```Bash
cd scripts/

CUDA_VISIBLE_DEVICES=1 python3 oas_optimization.py --task_id dfire --track_with_wandb True --wandb_entity nmaus --min_allowed_edit_dist 10 --num_initialization_points 10000 --max_string_length 200 --bsz 10 - run_lolbo - done 
```

# Arguments: 
## --optimize_pose 
To to allow slight tweaks to the aligned pose, add --optimize_pose True, it is false by default meaning that we just use the default pose optained by aligning to the parental antibody sequence with pymol 

## --min_allowed_edit_dist
Sets the minimum allowed edit distance allowed from the parental seq as a constraint
Set to -1 to allow any edit distance (no constraint)

## --num_initialization_points
Of the initization sequeces you provide in init_data.csv, this argument decides how many you want to actually use to bootstrap the optimization run (you can just set to the number of sequences in init_data.csv if you want to use all sequencess)

## --max_string_length
The maximum allowed length of any antibody sequence. Decrease this if you are running into OOM issues. 

## --task_id 
Specifies the objective, can be dfire, dfire2, or cpydock 

## --bsz
Batch size (number of sequences queried at once on each optimization step)



