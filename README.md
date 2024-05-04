# ZeroDDI
ZeroDDI: A Zero-Shot Drug-Drug Interaction Event Prediction Method with Semantic Enhanced Learning and Dual-Modal Uniform Alignment

## 1 install 
create a conda virtual env:
`conda create -n name python=3.8`

The required libraries are:

- numpy

- torch 1.11.0+cu113

- addict

- yapf

- sklearn

- pandas

- torch_geometric 2.1.0

- rdkit

- tensorflow

- deepchem

- networkx

- transformers>.4.26

- matplotlib

- sacremoses

- bs4

- lxml

## 2 Usage
### 2.1 Training ZeroDDI
`python main.py --config configs/zeroddi.py
`

### 2.2 Testing ZeroDDI
After training, the parameters of models are saved in ./work_dirs/
Then, you can test the model by:
`python main.py ---config configs/zeroddi.py --zsl_para work_dirs/zeroddi/model_parameter/zsl_model_best_epoch100_seed42.pkl --gzsl_para work_dirs/zeroddi/model_parameter/gzsl_model_best_epoch100_seed42.pkl `



