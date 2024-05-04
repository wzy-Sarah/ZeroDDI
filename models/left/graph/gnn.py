# -*- coding: utf-8 -*-
import warnings
from tools.registry import Registry
from ...builder import build_left, LEFT
from torch import nn
import torch
import numpy as np
import time
import pandas as pd
import os.path as osp
import pickle as pkl
import os
from rdkit.Chem import AllChem
from .graph_init import mol_to_graph_data_obj_simple
from torch_geometric.nn import global_mean_pool, MessagePassing
#from torch_geometric.nn import GATConv
# from torch_geometric.nn import global_mean_pool,MessagePassing,GCNConv,GINConv,SAGEConv,GATConv
from torch_geometric.data import Batch
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_scatter import scatter_add
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        norm = self.norm(edge_index[0], x.size(0), x.dtype)
        x = self.linear(x)
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__(node_dim=0)
        self.aggr = aggr
        self.heads = heads
        self.emb_dim = emb_dim
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(emb_dim, heads * emb_dim)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))
        self.bias = nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, heads * emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)

        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + \
                          self.edge_embedding2(edge_attr[:, 1])

        x = self.weight_linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out += self.bias
        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.gnn_type = gnn_type

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim, aggr="add"))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim,aggr="add"))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim, aggr="mean"))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 2:
            x, edge_index = argv[0], argv[1]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


@LEFT.register_module()
class GNN_model(nn.Module):
    def __init__(self, Allfilename, dropout, device, num_layer, JK, gnn_type, output_dim, extra_data=None):
        super().__init__()
        """
        gnn_type are: "gin","gcn","gat","graphsage"
        """
        self.Allfilename = Allfilename
        self.mol_dim = output_dim
        self.device = device
        self.extra_data = extra_data

        self.num_layer = num_layer
        self.dropout_ratio = dropout
        self.JK = JK
        self.gnn_type = gnn_type

        self.name2data = self.get_graph_data()
        self.gnn = GNN(self.num_layer, 300, JK=self.JK, drop_ratio=self.dropout_ratio, gnn_type=self.gnn_type)
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))
        self.linear = nn.Linear(600, self.mol_dim)
        self.Dropout_layer = nn.Dropout(self.dropout_ratio)
        

    def get_graph_data(self):
        
        dataset = pd.read_csv(self.Allfilename)
        drug1_list = [[a, b] for a, b in zip(dataset["drug1"], dataset["smiles1"])]
        drug2_list = [[a, b] for a, b in zip(dataset["drug2"], dataset["smiles2"])]
        drug_list = drug1_list + drug2_list
        
        if self.extra_data is not None:
            dataset1 = pd.read_csv(self.extra_data)
            drug1_list1 = [[a, b] for a, b in zip(dataset1["drug1"], dataset1["smiles1"])]
            drug2_list1 = [[a, b] for a, b in zip(dataset1["drug2"], dataset1["smiles2"])]
            drug_list1 = drug1_list1 + drug2_list1
            drug_list = drug_list + drug_list1
        self.drugname_smiles = {}
        for d in drug_list:
            self.drugname_smiles[d[0]] = d[1]
        data_dict = {}
        for name, smiles in self.drugname_smiles.items():
            # each example contains a single species
            try:
                rdkit_mol = AllChem.MolFromSmiles(smiles)
                if rdkit_mol != None:  # ignore invalid mol objects
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # manually add mol id
                    id = int(name.split('DB')[1].lstrip('0'))
                    # id = int(zinc_id_list[i].split('DB')[1].lstrip('0'))
                    data.id = torch.tensor(
                        [id])  # id here is zinc id value, stripped of
                    # leading zeros
                    data_dict[name] = data

            except:
                continue
        return data_dict

    def forward(self, inputs):
        # input : drug1,drug2,event_type
        t1 = time.time()
        drug1s = inputs[0]
        drug2s = inputs[1]
        drug1_list = []
        drug2_list = []
        for i in range(len(drug1s)):
            drug1 = drug1s[i]
            drug2 = drug2s[i]
            data1 = self.name2data[drug1]
            data2 = self.name2data[drug2]
            drug1_list.append(data1.to(self.device))
            drug2_list.append(data2.to(self.device))
        drug1_batch = Batch.from_data_list(drug1_list)
        drug2_batch = Batch.from_data_list(drug2_list)

        x1 = self.gnn(drug1_batch.x, drug1_batch.edge_index, drug1_batch.edge_attr)
        
        x1 = self.pool(x1, drug1_batch.batch)
        x1 = self.projection_head(x1)

        x2 = self.gnn(drug2_batch.x, drug2_batch.edge_index, drug2_batch.edge_attr)
        x2 = self.pool(x2, drug2_batch.batch)
        x2 = self.projection_head(x2)

        drugpair_feature = torch.cat((x1, x2), 1)
        out = self.linear(drugpair_feature)
        out = self.Dropout_layer(out)
        return out

@LEFT.register_module()
class CV_GNN_model(nn.Module):
    def __init__(self, Allfilename, dropout, device, num_layer, JK, gnn_type, output_dim, extra_data=None,data_dict=None):
        super().__init__()
        """
        gnn_type are: "gin","gcn","gat","graphsage"
        """
        self.Allfilename = Allfilename
        self.mol_dim = output_dim
        self.device = device
        self.extra_data = extra_data
        self.data_dict=data_dict

        self.num_layer = num_layer
        self.dropout_ratio = dropout
        self.JK = JK
        self.gnn_type = gnn_type

        #self.name2data = self.get_graph_data()
        self.gnn = GNN(self.num_layer, 300, JK=self.JK, drop_ratio=self.dropout_ratio, gnn_type=self.gnn_type)
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))
        self.linear = nn.Linear(600, self.mol_dim)
        self.Dropout_layer = nn.Dropout(self.dropout_ratio)

    
    def forward(self, inputs):
        # input : drug1,drug2,event_type
        
        drug1_batch=inputs[0]
        drug2_batch=inputs[1]

        x1 = self.gnn(drug1_batch.x, drug1_batch.edge_index, drug1_batch.edge_attr)

        x1 = self.pool(x1, drug1_batch.batch)
        x1 = self.projection_head(x1)

        x2 = self.gnn(drug2_batch.x, drug2_batch.edge_index, drug2_batch.edge_attr)
        x2 = self.pool(x2, drug2_batch.batch)
        x2 = self.projection_head(x2)

        drugpair_feature = torch.cat((x1, x2), 1)
        out = self.linear(drugpair_feature)
        out = self.Dropout_layer(out)
        return out
    
    
@LEFT.register_module()
class gen_GNN_model(nn.Module):
    def __init__(self, Allfilename, dropout, device, num_layer, JK, gnn_type, extra_data=None):
        super().__init__()
        """
        gnn_type are: "gin","gcn","gat","graphsage"
        """
        self.Allfilename = Allfilename
        self.mol_dim = 600
        self.device = device
        self.extra_data = extra_data

        self.num_layer = num_layer
        self.dropout_ratio = dropout
        self.JK = JK
        self.gnn_type = gnn_type

        self.name2data = self.get_graph_data()
        self.gnn = GNN(self.num_layer, 300, JK=self.JK, drop_ratio=self.dropout_ratio, gnn_type=self.gnn_type)
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))
        

    def get_graph_data(self):

        dataset = pd.read_csv(self.Allfilename)
        drug1_list = [[a, b] for a, b in zip(dataset["drug1"], dataset["smiles1"])]
        drug2_list = [[a, b] for a, b in zip(dataset["drug2"], dataset["smiles2"])]
        drug_list = drug1_list + drug2_list

        self.drugname_smiles = {}
        for d in drug_list:
            self.drugname_smiles[d[0]] = d[1]
        data_dict = {}
        for name, smiles in self.drugname_smiles.items():
            # each example contains a single species
            try:
                rdkit_mol = AllChem.MolFromSmiles(smiles)
                if rdkit_mol != None:  # ignore invalid mol objects
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # manually add mol id
                    id = int(name.split('DB')[1].lstrip('0'))
                    # id = int(zinc_id_list[i].split('DB')[1].lstrip('0'))
                    data.id = torch.tensor(
                        [id])  # id here is zinc id value, stripped of
                    # leading zeros
                    data_dict[name] = data

            except:
                continue
        return data_dict

    def forward(self, inputs):
        # input : drug1,drug2,event_type
        t1 = time.time()
      
        drug1_batch = inputs[0]
        drug2_batch = inputs[1]

        x1 = self.gnn(drug1_batch.x, drug1_batch.edge_index, drug1_batch.edge_attr)

        x1 = self.pool(x1, drug1_batch.batch)
        x1 = self.projection_head(x1)

        x2 = self.gnn(drug2_batch.x, drug2_batch.edge_index, drug2_batch.edge_attr)
        x2 = self.pool(x2, drug2_batch.batch)
        x2 = self.projection_head(x2)

        drugpair_feature = torch.cat((x1, x2), 1)
        # out = self.linear(drugpair_feature)
        # out = self.Dropout_layer(out)
        return drugpair_feature


    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)    
        
@LEFT.register_module()    
class Encoder(nn.Module):

    def __init__(self, encoder_layer_sizes,latent_size):

        super(Encoder,self).__init__()
        layer_sizes = encoder_layer_sizes
        latent_size = latent_size
        layer_sizes[0] += latent_size
        self.fc1=nn.Linear(layer_sizes[0], layer_sizes[-1])
        self.fc3=nn.Linear(layer_sizes[-1], latent_size*2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.linear_means = nn.Linear(latent_size*2, latent_size)
        self.linear_log_var = nn.Linear(latent_size*2, latent_size)
        self.apply(weights_init)

    def forward(self, x, c=None):
        if c is not None: x = torch.cat((x, c), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

@LEFT.register_module()      
class Generator(nn.Module):

    def __init__(self, opt):

        super(Generator,self).__init__(decoder_layer_sizes,latent_size)

        layer_sizes = decoder_layer_sizes
        latent_size = latent_size
        input_size = latent_size * 2
        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc3 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid=nn.Sigmoid()
        self.apply(weights_init)

    def _forward(self, z, c=None):
        z = torch.cat((z, c), dim=-1)
        x1 = self.lrelu(self.fc1(z))
        x = self.sigmoid(self.fc3(x1))
        self.out = x1
        return x

    def forward(self, z, a1=None, c=None, feedback_layers=None):
        if feedback_layers is None:
            return self._forward(z,c)
        else:
            z = torch.cat((z, c), dim=-1)
            x1 = self.lrelu(self.fc1(z))
            feedback_out = x1 + a1*feedback_layers
            x = self.sigmoid(self.fc3(feedback_out))
            return x    

        
#conditional discriminator for inductive
@LEFT.register_module()  
class Discriminator_D1(nn.Module):
    def __init__(self, opt): 
        super(Discriminator_D1, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        return h      
    
#Feedback Modules
@LEFT.register_module() 
class Feedback(nn.Module):
    def __init__(self,opt):
        super(Feedback, self).__init__()
        self.fc1 = nn.Linear(opt.ngh, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)
    def forward(self,x):
        self.x1 = self.lrelu(self.fc1(x))
        h = self.lrelu(self.fc2(self.x1))
        return h

@LEFT.register_module()    
class AttDec(nn.Module):
    def __init__(self, opt, attSize):
        super(AttDec, self).__init__()
        self.embedSz = 0
        self.fc1 = nn.Linear(opt.resSize + self.embedSz, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.hidden = None
        self.sigmoid = None
        self.apply(weights_init)

    def forward(self, feat, att=None):
        h = feat
        if self.embedSz > 0:
            assert att is not None, 'Conditional Decoder requires attribute input'
            h = torch.cat((feat,att),1)
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc3(self.hidden)
        if self.sigmoid is not None: 
            h = self.sigmoid(h)
        else:
            h = h/h.pow(2).sum(1).sqrt().unsqueeze(1).expand(h.size(0),h.size(1))
        self.out = h
        return h

    def getLayersOutDet(self):
        #used at synthesis time and feature transformation
        return self.hidden.detach()    

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

