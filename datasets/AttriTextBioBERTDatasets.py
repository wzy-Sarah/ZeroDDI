# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import numpy as np
from collections import defaultdict
from .builder import DATASETS, build_dataset
from transformers import AutoTokenizer, BioGptModel,AutoModel 
from .Mesh_similarity import MeshText

    
@DATASETS.register_module()
class AttriTextBioBERTDataset(Dataset):

    def __init__(self,
                 Allfilename,
                 mode,
                 file_dir,
                 file_name,
                 zsl_mode,
                 output_file=None,
                 output_dim=None,
                 #input_dim = None,
                 device='cuda:0',
                 bert_vision="biobert-base-cased-v1.2"
                 ):
      
        self.Allfilename = Allfilename
        self.mode = mode
        self.file_dir = file_dir
        self.file_name = file_name
        self.output_file = output_file
        self.device = device
        self.output_dim = output_dim
        self.zsl_mode = zsl_mode

        self.bert_vision = bert_vision
        self.mesh_text_biobertemb, self.biobertemb = self._get_all_embeddings()
        
        self.current_all_biogpt_emb, self.current_all_mesh_emb = self.get_current_dataset()
        self.rightinput = (self.current_all_biogpt_emb, self.current_all_mesh_emb)
        self.rightattributelabel = (self.current_sign_id, self.current_mesh_id, self.current_patt_id)
        self.dim = self.output_dim
        self.input_dim = (self.current_all_biogpt_emb.shape[2], self.current_all_mesh_emb.shape[2])

    def _get_all_embeddings(self):
        # id,drug1,drug2,event_id,MeSH_ID,Sign,Pattern,description,smiles1,smiles2
     
        df_all_dataset = pd.read_csv(self.Allfilename)
        all_dataset = [[id1, id2, ddi_type, a, b, c] for id1, id2, ddi_type, a, b, c in
                       zip(df_all_dataset['drug1'], df_all_dataset['drug2'],
                           df_all_dataset['event_id'],
                           df_all_dataset['MeSH_ID'], df_all_dataset['Sign'],
                           df_all_dataset['Pattern'])]

        print(f"The {self.Allfilename} dataset has {len(all_dataset)} DDIs.")

        self.all_triplet = {}
        for value in all_dataset:
            self.all_triplet[value[2]] = [value[3], value[4].strip(), value[5]]  # event_id to effect, sign, pattern

        self.Meshids = []
        self.Signs = []
        self.Pattern = []
        for key, value in self.all_triplet.items():
            mesh = value[0].split("&")  # because one sample may has more than one mesh id
            sign = value[1].strip().split("&")
            patt = value[2]
            for i in mesh:
                if i not in self.Meshids:
                    self.Meshids.append(i)
            for i in sign:
                if i not in self.Signs:
                    self.Signs.append(i)
            if patt not in self.Pattern:
                self.Pattern.append(patt)

        evenidanddes = [[a, b] for a, b in zip(df_all_dataset['event_id'], df_all_dataset['description'])]
        self.all_descriptions = []
        self.eventid = []
        for item in evenidanddes:
            id, d = item[0], item[1]
            if d not in self.all_descriptions:
                self.all_descriptions.append(d)
                self.eventid.append(id)
        assert len(self.eventid) == len(self.all_descriptions)
        print(f"The {self.Allfilename} dataset has {len(self.all_descriptions)} descriptions")

        file = os.path.join(self.output_file, f"{self.bert_vision}_mesh_text_embedding.pt")
        if os.path.exists(file):
            mesh_text_biobertemb = torch.load(file)
            #print("mesh_text_biobertemb",mesh_text_biobertemb.shape)
        else:
            mesh = MeshText(self.Meshids)
            mesh_text = mesh.get_text()

            tokenizer = AutoTokenizer.from_pretrained(f"data/{self.bert_vision}") #f"dmis-lab/{self.bert_vision}"
            model = AutoModel.from_pretrained(f"data/{self.bert_vision}")
            inputs = tokenizer(mesh_text, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            mesh_text_biobertemb = outputs.last_hidden_state
            #print("mesh_text_biobertemb", mesh_text_biobertemb.shape)#[114,131,768]
            torch.save(mesh_text_biobertemb, file)
            # df_disease_sim = pd.DataFrame(mesh_smi)
            # df_disease_sim.index = self.Meshids
            # df_disease_sim.to_csv(file, sep=',', )
        #mesh_text_biobertemb = torch.mean(mesh_text_biobertemb,1)
        print("mesh_text_biobertemb",mesh_text_biobertemb.shape,type(mesh_text_biobertemb))

        biobertembfile = os.path.join(self.output_file, f"{self.bert_vision}2.pt")


        if not os.path.exists(biobertembfile):
            tokenizer = AutoTokenizer.from_pretrained(f"data/{self.bert_vision}")
            model = AutoModel.from_pretrained(f"data/{self.bert_vision}")
            inputs = tokenizer(self.all_descriptions, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            biobertemb = outputs.last_hidden_state
            print("biogptemb", biobertemb.shape)
            torch.save(biobertemb, biobertembfile)
        else:
            biobertemb = torch.load(biobertembfile)  # []
        return mesh_text_biobertemb, biobertemb  # attrionehot,biogptemb

    def get_current_dataset(self):
    
        df_dataset = pd.read_csv(os.path.join(self.file_dir, self.file_name))
        current_dataset = [[id1, id2, ddi_type] for id1, id2, ddi_type in
                           zip(df_dataset['drug1'], df_dataset['drug2'],
                               df_dataset['event_id'])]

        print(f"The {self.file_name} dataset has {len(current_dataset)} DDIs.")


        current_dataset_eventids = list(df_dataset['event_id'])

        self.current_dataset_eventid_uni = []
        for i in current_dataset_eventids:
            if i not in self.current_dataset_eventid_uni:
                self.current_dataset_eventid_uni.append(i)
  
        self.eventid2embid = {}
        self.embid2eventid = {}
        count = 0
        mesh_current = []
        self.current_sign_id=[]
        self.current_mesh_id=[]
        self.current_patt_id=[]

        for type in self.current_dataset_eventid_uni:
            item = self.all_triplet[type]
            mesh = item[0].split("&")
            effect_emb = []
            for meshid in mesh:
                id = self.Meshids.index(meshid)
                meshtext = self.mesh_text_biobertemb[id,:].detach().numpy()
                effect_emb.append(meshtext)
                #id = self.Meshids.index(i)
                #meshids.append(id)
            mesh_emb = np.array(effect_emb)
            mesh_emb_mean = np.mean(mesh_emb,0)
            mesh_current.append(mesh_emb_mean)

            sign = item[1].strip().split("&")[0]
            sign_id = self.Signs.index(sign)
            self.current_sign_id.append(sign_id)

            effect = mesh[0]
            effect_id = self.Meshids.index(effect)
            self.current_mesh_id.append(effect_id)

            pattern = item[2]
            pattern_id = self.Pattern.index(pattern)
            self.current_patt_id.append(pattern_id)

            self.eventid2embid[type] = count
            self.embid2eventid[count] = type
            count = count + 1
    
        ind = [self.eventid.index(i) for i in self.current_dataset_eventid_uni]
   
        current_all_biogpt_emb = self.biobertemb[ind, :, :].to(self.device)

        current_all_mesh_emb = torch.tensor(np.array(mesh_current), dtype=torch.float32).to(self.device)

       
        self.new_current_dataset = []
        for item in current_dataset:
            embid = self.eventid2embid[int(item[2])]
            self.new_current_dataset.append([item[0], item[1], embid])
        return current_all_biogpt_emb, current_all_mesh_emb

    def __getitem__(self, index):
        return (
        self.new_current_dataset[index], self.mode,self.zsl_mode)

    def __len__(self):
        return len(self.new_current_dataset)

