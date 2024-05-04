import rdkit
from rdkit import Chem
import deepchem as dc


drug_feature = {} 
featurizer = dc.feat.ConvMolFeaturizer() 
for tup in zip(drug['pubchem'], drug['isosmiles']):
    mol=Chem.MolFromSmiles(tup[1])
    X = featurizer.featurize(mol)
    drug_feature[str(tup[0])]=[X[0].get_atom_features(),X[0].get_adjacency_list()]


def CalculateGraphFeat(feat_mat,adj_list):
    use_molecular_graph = True
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
    if use_molecular_graph==True:
        for i in range(len(adj_list)):
            nodes = adj_list[i]
            for each in nodes:
                adj_mat[i,int(each)] = 1
        assert np.allclose(adj_mat,adj_mat.T)
    else:
        adj_mat = adj_mat + np.eye(len(adj_list))
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    return [feat_mat,adj_index]


def FeatureExtract(drug_feature):
    drug_data = [[] for item in range(len(drug_feature))]
    for i in range(len(drug_feature)):
        feat_mat, adj_list = drug_feature.iloc[i]
        drug_data[i] = CalculateGraphFeat(feat_mat,adj_list)
    return drug_data


drug_data = FeatureExtract(drug_feature)


drug_set = Data.DataLoader(dataset=GraphDataset(graphs_dict=drug_data),collate_fn=collate,batch_size=nb_drugs,shuffle=False)



