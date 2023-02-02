import numpy as np
import torch 
import sys 
sys.path.append("../")
sys.path.append("../../dockbo/")
from oas_heavy_ighg_vae.transformer_vae_unbounded import InfoTransformerVAE
from oas_heavy_ighg_vae.data import collate_fn, DataModuleKmers
from lolbo.latent_space_objective import LatentSpaceObjective
from seq_to_score import seqs_to_scores, compute_edit_distance
from igfold import IgFoldRunner, init_pyrosetta 
from dockbo.dockbo import DockBO
from constants import (
    PATH_TO_VAE_STATE_DICT, 
    PATH_TO_ANTIGEN_PDB, 
    WORK_DIR,
    PARENTAL_H_CHIAN
)

class OasObjective(LatentSpaceObjective):
    '''OasObjective class supports all antibody IGIH heavy chain
         optimization tasks and uses the OAS VAE by default '''

    def __init__(
        self,
        task_id='dfire',
        path_to_vae_statedict=PATH_TO_VAE_STATE_DICT,
        xs_to_scores_dict={},
        max_string_length=1024,
        num_calls=0,
        # min_length_constraint=None,
        # max_length_constraint=10,
        anitgen_pdb_path=PATH_TO_ANTIGEN_PDB,
        work_dir=WORK_DIR,
        optimize_pose=False,
        min_allowed_edit_dist=-1,
    ):
        assert task_id in ["dfire", "dfire2", "cpydock"]
        self.dim                    = 256 # SELFIES VAE DEFAULT LATENT SPACE DIM
        self.path_to_vae_statedict  = path_to_vae_statedict # path to trained vae stat dict
        self.max_string_length      = max_string_length # max string length that VAE can generate
        # self.min_length_constraint  = min_length_constraint
        # self.max_length_constraint  = max_length_constraint 
        self.optimize_pose          = optimize_pose

        # minimum allowed number of edits from the parental h chian sequence 
        #   * set MIN_ALLOWED_EDIT_DIST = -1 to allow any number of edits 
        self.min_allowed_edit_dist = min_allowed_edit_dist

        init_pyrosetta()
        self.igfold_runner = IgFoldRunner()
        self.oracle = DockBO( 
            path_to_default_antigen_pdb=anitgen_pdb_path,
            work_dir=work_dir, 
            scoring_func=task_id,
            init_bo_w_known_pose=True, # initialize pose optimization w/ known pose
            verbose_config_opt=False, # print updates during pose optimization
            # bo args only relevant if self.optimize_pose = True
            max_n_bo_steps=100,
            absolute_max_n_steps=300,
            bsz=1,
            n_init=10,
            init_n_epochs=20,
            update_n_epochs=2,
        )

        super().__init__(
            num_calls=num_calls,
            xs_to_scores_dict=xs_to_scores_dict,
            task_id=task_id,
        )


    def vae_decode(self, z):
        '''Input
                z: a tensor latent space points
            Output
                a corresponding list of the decoded input space 
                items output by vae decoder 
        '''
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).float()
        z = z.cuda()
        self.vae = self.vae.eval()
        self.vae = self.vae.cuda()
        # sample molecular string form VAE decoder
        sample = self.vae.sample(z=z.reshape(-1, 2, 128))
        # grab decoded aa strings
        decoded_seqs = [self.dataobj.decode(sample[i]) for i in range(sample.size(-2))]

        return decoded_seqs


    def query_oracle(self, x):
        ''' Input: 
                list of items x (list of aa seqs)
            Output:
                method queries the oracle and returns 
                the corresponding score y,
                or np.nan in the case that x is an invalid input
                for each item in input list
        '''
        # method assumes x is a single smiles string
        scores = seqs_to_scores(
            seqs=x,
            igfold_runner=self.igfold_runner,
            oracle=self.oracle,
            optimize_pose=self.optimize_pose,
        )
        # scores = []
        # for seq in x:
        #     energy = seq_to_score(
        #         aa_seq=seq,
        #         igfold_runner=self.igfold_runner,
        #         oracle=self.oracle,
        #         optimize_pose=self.optimize_pose,
        #     )
        #     scores.append(energy)
        # scores = [len(seq) for seq in x]
        # score = smiles_to_desired_scores([x], self.task_id, tdc_oracle=self.tdc_oracle).item()

        return scores


    def initialize_vae(self):
        ''' Sets self.vae to the desired pretrained vae and 
            sets self.dataobj to the corresponding data class 
            used to tokenize inputs, etc. '''
        # self.dataobj = SELFIESDataset()

        self.dataobj = DataModuleKmers(
            batch_size=10,
            k=3,
            seq_len_upper_bound=800,
            seq_len_lower_bound=10,
        )
        self.dataobj = self.dataobj.train
        self.vae = InfoTransformerVAE(dataset=self.dataobj, d_model=128)
        # load in state dict of trained model:
        if self.path_to_vae_statedict:
            state_dict = torch.load(self.path_to_vae_statedict) 
            self.vae.load_state_dict(state_dict, strict=True) 
        self.vae = self.vae.cuda()
        self.vae = self.vae.eval()
        # set max string length that VAE can generate
        self.vae.max_string_length = self.max_string_length

    def vae_forward(self, xs_batch):
        ''' Input: 
                a list xs 
            Output: 
                z: tensor of resultant latent space codes 
                    obtained by passing the xs through the encoder
                vae_loss: the total loss of a full forward pass
                    of the batch of xs through the vae 
                    (ie reconstruction error)
        '''
        # assumes xs_batch is a batch of smiles strings 
        tokenized_seqs = self.dataobj.tokenize_sequence(xs_batch)
        encoded_seqs = [self.dataobj.encode(seq).unsqueeze(0) for seq in tokenized_seqs]
        X = collate_fn(encoded_seqs)
        dict = self.vae(X.cuda())
        vae_loss, z = dict['loss'], dict['z']
        z = z.reshape(-1,self.dim)

        return z, vae_loss


    def compute_constraints(self, xs_batch):
        ''' Input: 
                a list xs 
            Output: 
                c: tensor of size (len(xs),n_constraints) of
                    resultant constraint values, or
                    None of problem is unconstrained
                    Note: constraints, must be of form c(x) <= 0!
        '''
        constraints_list = []
        if self.min_allowed_edit_dist > 0:
            c_vals1 = []
            for x in xs_batch:
                edit_dist = compute_edit_distance(x, PARENTAL_H_CHIAN)
                c_vals1.append(edit_dist - self.min_allowed_edit_dist) 
            c_vals1 = torch.tensor(c_vals1).unsqueeze(-1)
            constraints_list.append(c_vals1)
        
        if len(constraints_list) > 0:
            return torch.cat(constraints_list, dim=-1)
        else:
            return None 


if __name__ == "__main__":
    # testing molecule objective
    obj1 = OasObjective(task_id='pdop' ) 
    print(obj1.num_calls)
    dict1 = obj1(torch.randn(10,256))
    print(dict1['scores'], obj1.num_calls)
    dict1 = obj1(torch.randn(3,256))
    print(dict1['scores'], obj1.num_calls)
    dict1 = obj1(torch.randn(1,256))
    print(dict1['scores'], obj1.num_calls)
