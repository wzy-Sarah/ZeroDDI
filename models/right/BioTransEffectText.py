# -*- coding:utf-8 -*-
import sys
import torch
import numpy as np
import pandas as pd
import os.path as osp
from ..builder import build_left, RIGHT
import requests
import warnings
import torch.nn as nn

# from .Mesh_similarity import MeshSimilarity
from torch import nn

# from tools.utils import mkdir_or_exist
warnings.filterwarnings("ignore")  # remove the warning
import os

os.environ['NO_PROXY'] = 'nlm.nih.gov'
import pickle
from sklearn.preprocessing import OneHotEncoder

SIGNS = ['increase', 'decrease']





@RIGHT.register_module()
class BioTransEffectTextttention(nn.Module):
    def __init__(self, output_dim=None, input_dim=None, device=None):
        super().__init__()
        self.input_dim = input_dim
        input_bio_dim = self.input_dim[0]
        input_sim_dim = self.input_dim[1]
        self.output_dim = output_dim
        self.encoder1 = nn.Sequential(
            nn.Linear(input_bio_dim, input_bio_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(input_bio_dim // 2, input_bio_dim // 2//2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(input_bio_dim // 2//2,self.output_dim),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(input_sim_dim, input_bio_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(input_bio_dim // 2, input_bio_dim // 2//2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(input_bio_dim // 2//2, self.output_dim ),
        )
        self.layernorm = nn.LayerNorm(self.output_dim)
        self.linear =nn.Sequential( nn.Linear(self.output_dim, self.output_dim),
                                   nn.LeakyReLU()
                                  )

    def forward(self, inputsall,emb):
        current_all_biobert_emb = emb[0].data
        current_all_biobert_emb1 = self.encoder1(current_all_biobert_emb)
        current_all_mesh_emb = emb[1].data
        mean = torch.mean(current_all_mesh_emb,1)
        current_all_mesh_emb1 = self.encoder2(mean)
        con = torch.cat((current_all_biobert_emb1,current_all_mesh_emb1.unsqueeze(1)),dim=1)
        con = self.layernorm(con)
        con = self.linear(con)
        return con, inputsall[2], current_all_biobert_emb1
    



