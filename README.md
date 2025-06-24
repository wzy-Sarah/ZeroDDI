# ZeroDDI
ZeroDDI: A Zero-Shot Drug-Drug Interaction Event Prediction Method with Semantic Enhanced Learning and Dual-Modal Uniform Alignment

Accepted by IJCAI2024

The authors are Ziyan Wang, Zhankun Xiong, Feng Huang, Xuan Liu, Wen Zhang.

## 1 install 
create a conda virtual env:
`conda env create -f environment.yaml`

Required libraries are:

numpy

torch 1.11.0+cu113

addict

yapf

sklearn

pandas

torch_geometric 2.1.0

rdkit

tensorflow

deepchem

networkx

transformers>=4.26

matplotlib

sacremoses

bs4

lxml

## 2 Usage
### 2.1 dataset
Due to space limitations, we compressed the dataset. You can unzip all xxx.zip data in its fold.

`cd data/DrugBank5.1.9/`

`unzip DDI_final.zip -d .`

`cd zsl`

`unzip train.zip -d .`

`cd gzsl`

`unzip train.zip -d .`

### 2.2 checkpoints
Download the biobert checkpoints from the huggingface [
biobert-base-cased-v1.2/](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2/tree/main)
And put it to data/

### 2.3 Training ZeroDDI
There are three folds in /data/DrugBank5.1.9/ 

For example, the data of fold2 is in zsl2/ and gzsl2/

`python main.py --config configs/zeroddi.py
`
or `python main.py --config configs/zeroddi_fold2.py
`

You can also create our own config python file for different datasets or models.


### 2.4 Testing ZeroDDI
After training, the parameters of models are saved in ./work_dirs/

Then, you can test the model by:

`python main.py --config configs/zeroddi.py --zsl_para work_dirs/zeroddi/model_parameter/zsl_model_best_epoch100_seed42.pkl --gzsl_para work_dirs/zeroddi/model_parameter/gzsl_model_best_epoch100_seed42.pkl `

### 2.5 Cite Us
```bibtex
@inproceedings{wang2024zeroddi,
  title={ZeroDDI: a zero-shot drug-drug interaction event prediction method with semantic enhanced learning and dual-modal uniform alignment},
  author={Wang, Ziyan and Xiong, Zhankun and Huang, Feng and Liu, Xuan and Zhang, Wen},
  booktitle={Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence},
  pages={6071--6079},
  year={2024}
}

