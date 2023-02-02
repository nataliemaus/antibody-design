import sys 
sys.path.append("../")
import pymolPy3
import os 
import uuid 
from Bio.PDB import PDBParser, PDBIO, Select
from constants import (
    KNOWN_AB_POSE_PATH, 
    LIGHT_CHAIN,
    WORK_DIR,
    REPO_NAME
)
import numpy as np 
import wandb 
import pandas as pd 

def align(
    save_aligned_pdb_path,
    ab_path,
    ab_name
): 
    # ** MUST RELOAD EACH TIME OR BREAKS! 
    pm = pymolPy3.pymolPy3(0)
    known_pose_file_name = KNOWN_AB_POSE_PATH.split("/")[-1].split(".")[-2]
    pm('delete all') 
    pm(f'load {KNOWN_AB_POSE_PATH}')
    pm(f'load {ab_path}')
    pm(f'super {ab_name}, {known_pose_file_name}')
    pm(f'select old_ab, model {known_pose_file_name}') # you'll have to figure out which chain is which
    pm(f'remove old_ab') 
    pm(f"save {save_aligned_pdb_path}") 
    pm('delete all') 

class NonHetSelect(Select):
    def accept_residue(self, residue):
        return 1 if residue.id[0] == " " else 0

def remove_hetero_atoms_and_hydrogens(path_to_pdb):
    # First, remove HET atoms
    pdb = PDBParser().get_structure(path_to_pdb.replace(".pdb", ""), path_to_pdb)
    io = PDBIO()
    io.set_structure(pdb)
    save_path = path_to_pdb.replace(".pdb", "_no_het.pdb")
    io.save(save_path, NonHetSelect())
    # Next, use pdb-tools to delete Hydrogen atoms 
    #   Need pip install pdb-tools
    #   http://www.bonvinlab.org/pdb-tools/
    save_path_noh = save_path.replace(".pdb", "_noh.pdb")
    os.system(f"pdb_delelem -H {save_path} > {save_path_noh}")
    os.remove(save_path) # remove temp no_het pdb file 

    return save_path_noh 


def fold_seq(
    heavy_chain_seq,
    save_path,
    igfold_runner,
): 
    sequences = {"H":heavy_chain_seq }
    sequences["L"] = LIGHT_CHAIN  # stays constant 
    out = igfold_runner.fold(
        save_path, # Output PDB file 
        sequences=sequences, # Antibody sequences
        do_refine=True, # Refine the antibsody structure with PyRosetta
        do_renum=False, # Renumber predicted antibody structure (Chothia) :) ! 
    )  ## do_renum=True causes bug :( 

    return save_path 

def fold_and_align(
    aa_seq,
    igfold_runner,
):
    ab_id = str(uuid.uuid1())
    folded_ab_path = WORK_DIR + REPO_NAME + f"/temp/{ab_id}.pdb"
    fold_seq(heavy_chain_seq=aa_seq, save_path=folded_ab_path, igfold_runner=igfold_runner)
    aligned_ab_path = WORK_DIR + REPO_NAME + f"/temp/{ab_id}_aligned.pdb"
    align(save_aligned_pdb_path=aligned_ab_path,ab_path=folded_ab_path, ab_name=ab_id)
    # remove hetero and hytrogens no work :( ... ends up w/ empy pdb after noh...
    aligned_ab_path_noh = remove_hetero_atoms_and_hydrogens(aligned_ab_path)
    # remove unnessary files created along the way... 
    os.remove(folded_ab_path)
    os.remove(aligned_ab_path)
    os.remove(folded_ab_path.replace(".pdb", ".fasta"))
    return aligned_ab_path_noh


def seq_to_score(
    aa_seq,
    igfold_runner,
    oracle,
    optimize_pose=False,
    remove_generated_pdbs=False,
):
    aa_seq = aa_seq.replace("-", "") # remove extra - tokens at end
    aligned_ab_path = fold_and_align(aa_seq, igfold_runner)

    if optimize_pose: # optimize pose to maximize score 
        config_x = None 
    else: # use the pose of the input pdb file and just compute the score 
        config_x = "default" 
    aligned_ab_path = str(aligned_ab_path)
    return_dict = oracle(
        config_x=config_x, 
        path_to_antibody_pdb=aligned_ab_path,
    )
    # remove pdb file after computing score to save space 
    if remove_generated_pdbs:
        os.remove(aligned_ab_path)

    return return_dict['energy']

def seqs_to_scores(
    seqs,
    igfold_runner,
    oracle,
    optimize_pose=False,
):
    # TODO spread this across multiple CPU in parallel... 
    scores = []
    for seq in seqs:
        try:
            energy = seq_to_score(
                aa_seq=seq,
                igfold_runner=igfold_runner,
                oracle=oracle,
                optimize_pose=optimize_pose,
            )
        except:
            energy = np.nan 
        scores.append(energy)
    return scores 


def compute_edit_distance(s1, s2): 
    ''' Returns Levenshtein Edit Distance btwn two strings'''
    m=len(s1)+1
    n=len(s2)+1
    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1 
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[i,j] 


# conda activate og_lolbo_mols 
# tmux attach -t run   !!! 
# CUDA_VISIBLE_DEVICES=0 python3 oas_optimization.py --task_id dfire --track_with_wandb True --wandb_entity nmaus --min_allowed_edit_dist 40 --num_initialization_points 20 --max_string_length 200 --bsz 2 - run_lolbo - done 
# actually working (getting scores of 2, 2, 8, ... )
# tmux attach -t run2 
# CUDA_VISIBLE_DEVICES=1 python3 oas_optimization.py --optimize_pose True --task_id dfire --track_with_wandb True --wandb_entity nmaus --min_allowed_edit_dist 40 --num_initialization_points 20 --max_string_length 200 --bsz 2 - run_lolbo - done 
# ^^ now running second one where we allow small pose opt (see if scores increaase now :) ... ) 
# current env saved as text file here: 
#   conda list --explicit > opt_ab.txt
# TO CREATE: 
# conda create --name opt_ab --file opt_ab.txt


# conda activate pymol  
# tmux attach -t run2   !!! 
# CUDA_VISIBLE_DEVICES=1 python3 oas_optimization.py --task_id dfire --track_with_wandb True --wandb_entity nmaus --min_allowed_edit_dist 20 --num_initialization_points 20 --max_string_length 200 --bsz 3 - run_lolbo - done 
# ** seems to be running significantly faster, not sure if that's because it fails faster tho or what
# JK, all zero on null scores, garbage file
# KILLED! 

# TODO: try allowing dockbo to take optimization steps to see if we get non-zero scores! ... 