import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F 
import pandas as pd
import itertools
import numpy as np
import json
import glob


class DataModuleKmers(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        k,
        seq_len_upper_bound=600,
        seq_len_lower_bound=20,
    ): 
        super().__init__() 
        self.batch_size = batch_size 
        self.k = k
        self.seq_len_upper_bound = seq_len_upper_bound
        self.seq_len_lower_bound = seq_len_lower_bound

        self.set_vocab()
        self.vocab2idx = { v:i for i, v in enumerate(self.vocab) }

        self.train  = DatasetKmers(
            dataset='train',
            k=self.k,
            vocab=self.vocab,
            vocab2idx=self.vocab2idx,
        )

        self.val    = DatasetKmers(
            dataset='val',
            k=k,
            vocab=self.train.vocab,
            vocab2idx=self.train.vocab2idx,
        )

        self.test   = DatasetKmers(
            dataset='test',
            k=k,
            vocab=self.train.vocab,
            vocab2idx=self.train.vocab2idx,
        )
    

    def set_vocab(self,):
        kmer_vocab_path = f'../oas_heavy_ighg_vae/vocab/{self.k}mer_vocab.csv'
        try:
            self.vocab = pd.read_csv(kmer_vocab_path, header=None).values.squeeze().tolist() 
        except: 
            amino_acids = pd.read_csv("../oas_heavy_ighg_vae/vocab/amino_acids.csv", header=None)
            amino_acids = amino_acids.values.squeeze().tolist() 
            amino_acids = set(amino_acids)
            self.vocab = ["".join(kmer) for kmer in itertools.product(amino_acids, repeat=self.k)] # 21**k tokens 
            self.vocab = ['<start>', '<stop>', *sorted(list(self.vocab))] # 21**k + 2 tokens 
            # save for next time 
            kmer_vocab_arr = np.array(self.vocab)
            pd.DataFrame(kmer_vocab_arr).to_csv(kmer_vocab_path, header=None, index=None)


    def make_seqs_kmers(self, regular_seqs):
        kmers = []
        for seq in regular_seqs:
            token_num = 0
            kmer_tokens = []
            while token_num < len(seq):
                kmer = seq[token_num:token_num+self.k]
                while len(kmer) < self.k:
                    kmer += '-' # padd so we always have length k 
                kmer_tokens.append("".join(kmer)) 
                token_num += self.k 
            kmers.append(kmer_tokens) 
        return kmers

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=10)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=10)



class DatasetKmers(Dataset): 
    def __init__(
        self,
        vocab,
        vocab2idx,
        dataset='train',
        k=3,
        data=[]
    ):
        self.dataset = dataset
        self.k = k
        self.vocab = vocab
        self.vocab2idx=vocab2idx
        self.data = data

    def tokenize_sequence(self, list_of_sequences):   
        ''' 
        Input: list of sequences in standard form (ie 'AGYTVRSGCMGA...')
        Output: List of tokenized sequences where each tokenied sequence is a list of kmers
        '''
        tokenized_sequences = []
        for seq in list_of_sequences:
            token_num = 0
            kmer_tokens = []
            while token_num < len(seq):
                kmer = seq[token_num:token_num + self.k]
                while len(kmer) < self.k:
                    kmer += '-' # padd so we always have length k  
                if type(kmer) == list: kmer = "".join(kmer)
                kmer_tokens.append(kmer) 
                token_num += self.k 
            tokenized_sequences.append(kmer_tokens) 
        return tokenized_sequences 

    def encode(self, tokenized_sequence):
        encoded_seq = []
        for s1 in [*tokenized_sequence, '<stop>']:
            if "*" in s1:
                s1 = s1.replace("*", "-")
            if "Z" in s1:
                s1 = s1.replace("Z", "-") # in ighgs from other species
            if "B" in s1:
                s1 = s1.replace("B", "-") # in ighgs from other species
            try:
                idx = self.vocab2idx[s1]
            except: 
                print(f'FAILURE ON tokenized_sequence {tokenized_sequence}')
                print(f'failure specifically on sequence: {s1}')
                raise RuntimeError('Failure to encode')
            encoded_seq.append(idx)
        encoded_seq = torch.tensor(encoded_seq)

        return encoded_seq


    def decode(self, tokens):
        '''
        Inpput: Iterable of tokens specifying each kmer in a given protien (ie [3085, 8271, 2701, 2686, ...] )
        Output: decoded protien string (ie GYTVRSGCMGA...)
        '''
        dec = [self.vocab[t] for t in tokens]
        # Chop out start token and everything past (and including) first stop token
        stop = dec.index("<stop>") if "<stop>" in dec else None # want first stop token
        protien = dec[0:stop] # cut off stop tokens
        while "<start>" in protien: # start at last start token (I've seen one case where it started w/ 2 start tokens)
            start = (1+dec.index("<start>")) 
            protien = protien[start:]
        protien = "".join(protien) # combine into single string 
        return protien

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encode(self.data[idx]) 

    @property
    def vocab_size(self):
        return len(self.vocab)


def collate_fn(data):
    # Length of longest molecule in batch 
    max_size = max([x.shape[-1] for x in data])
    return torch.vstack(
        # Pad with stop token
        [F.pad(x, (0, max_size - x.shape[-1]), value=1) for x in data]
    )



def translate(seq):
    table = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                 
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
    }
    protein =""
    for i in range(0, len(seq), 3):
        codon = seq[i:i + 3]
        protein += table[codon]
    return protein