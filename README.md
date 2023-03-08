# constrained-lolbo-for-antibodies
Code to run constrained lolbo for ab design (default is VAE pre-trained on OAS IGHG heavy chains)

# Getting Started 

## dockbo 
Make sure you have cloned dockbo in the same folder as you clone this repo. This repo uses dockbo to compute dfire, dfire2 and cpydock scores used to optimize:

https://github.com/nataliemaus/dockbo

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

CUDA_VISIBLE_DEVICES=1 python3 oas_optimization.py --task_id dfire --track_with_wandb True --wandb_entity nmaus --max_allowed_edit_dist 10 --num_initialization_points 10000 --max_string_length 200 --bsz 10 - run_lolbo - done 
```

# Arguments: 
## --optimize_pose 
To to allow slight tweaks to the aligned pose, add --optimize_pose True, it is false by default meaning that we just use the default pose optained by aligning to the parental antibody sequence with pymol 

## --max_allowed_edit_dist
Sets the maximum allowed edit distance allowed from the parental seq as a constraint


## --num_initialization_points
Of the initization sequeces you provide in init_data.csv, this argument decides how many you want to actually use to bootstrap the optimization run (you can just set to the number of sequences in init_data.csv if you want to use all sequencess)

## --max_string_length
The maximum allowed length of any antibody sequence. Decrease this if you are running into OOM issues. 

## --task_id 
Specifies the objective, can be dfire, dfire2, or cpydock 

## --bsz
Batch size (number of sequences queried at once on each optimization step)


# Optimizing a Diverse Population of Sequences
To generate a populartion of sequences rather than a single optimal sequence, 
we use the ROBOT algorithm from our paper recently accepted to AISTATs:
Discovering Many Diverse Solutions with Bayesian Optimization (ROBOT)
https://arxiv.org/abs/2210.10953

The ROBOT algorithm generates a set of M soltuions (M sequences), 
each of which optimize the objective. Additionally, ROBOT requires
that each of the M optimal sequences meet some threshold level of
diversity from eacother. In particular, we use edit distance and require
that each of the M optimal sequences has some threshold minimum edit distance
of tau from each of the other M sequences. 


## How to run ROBOT: 
```Bash
cd robot_script/

CUDA_VISIBLE_DEVICES=0 python3 diverse_oas_optimization.py --task_id dfire \
--max_n_oracle_calls 5000000 --bsz 10 \
--track_with_wandb True --wandb_entity nmaus \
--max_allowed_edit_dist 30 --num_initialization_points 10000 \
--max_string_length 60 --M 10 --tau 5 --save_csv_frequency 10 - run_robot - done 
```

## Additional Arguments (All other args are the same as for regular LOL-BO above): 
tmux attach -t test_robot 
conda activate og_lolbo_mols

### --M
The number of optimal sequences we seek to find 

### --tau 
The minimum threshold level of diversity required between the M optimal sequences
(here we use edit distance, so with --tau 5 we require the M sequences to each 
have a minimum edit distance of 5 from eacother)
Note that this is different from the requirement that each of the M 
sequences are also a MAXIMUM edit distance from the parental. 
i.e. with --max_allowed_edit_dist 20, --tau 5, we require that 
each of the M sequences is within 20 edits from the parental 
but also at least 5 edits from the other optimal sequences 




