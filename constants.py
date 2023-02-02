''' Constants used by optimization routine '''

# path to constnat light chain region seq (used by igfold)
LIGHT_CHAIN = "DIQLTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSTVPWTFGQGTKVEIK"

# parental heavy chain sequence (used to send minimum allowed edit distances)
PARENTAL_H_CHIAN = "EVQLVESGGGLVQPGGSLRLSCAASGYDFTHYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAVYYCAKYPYYYGTSHWYFDVWGQGTLVTVSS"

# curreent working directory (directory where constrain-lolbo-for_antibodies is cloned)
WORK_DIR = "/home/nmaus/"

# name of repo 
REPO_NAME = "constrained-lolbo-for-antibodies"

# path to state dict of OAS VAE
PATH_TO_VAE_STATE_DICT = WORK_DIR + REPO_NAME + "/oas_heavy_ighg_vae/saved_models/likely-deluge-195_model_state.pkl"

# path to csv file with headers seq, task_id1, task_id2, ... 
#   where the seq column has example sequences and 
#   i.e. task_id1 = dfire has dfire scorese for each sequence
PATH_TO_INITILIZATION_DATA = WORK_DIR + REPO_NAME + "/init_data.csv"

# path to pdb file with the known antibody we use to align each new seq to:
KNOWN_AB_POSE_PATH = WORK_DIR + REPO_NAME + "/pdbs/known_ab_pose.pdb"

# path to pdb file with the antigen we seek to bind to 
PATH_TO_ANTIGEN_PDB = WORK_DIR + REPO_NAME + "/pdbs/known_ag_pose.pdb"